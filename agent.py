import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 设置 HuggingFace 镜像加速

import uuid
import sqlite3
from typing import TypedDict, Annotated, Dict, Any, Optional, Union, List
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, RemoveMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from agent_config import MemoryAgentConfig


class State(TypedDict):
    """Agent状态定义"""
    messages: Annotated[list, add_messages]
    search_query: str
    retrieved_context: str


class MemoryAgent:
    """具有长期记忆功能的对话Agent"""
    
    def __init__(self, config: Optional[MemoryAgentConfig] = None):
        """
        初始化MemoryAgent
        
        Args:
            config: Agent配置,如为None则使用默认配置
        """
        self.config = config or MemoryAgentConfig()
        self._initialize_components()
        self._build_graph()
        
    def _initialize_components(self):
        """初始化所有组件"""
        # 初始化词向量嵌入模型
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config.embedding_model
        )
        
        # 初始化向量数据库
        self.vector_store = Chroma(
            collection_name="long_term_memory",
            embedding_function=self.embeddings,
            persist_directory=self.config.vector_store_path
        )
        
        # 初始化LLM
        self.llm = ChatOpenAI(
            model=self.config.llm_model,
            temperature=self.config.llm_temperature,
            openai_api_key=self.config.llm_api_key,
            openai_api_base=self.config.llm_api_base,
        )
        
        # 初始化检查点数据库连接
        self.connection = sqlite3.connect(
            self.config.checkpoints_db,
            check_same_thread=False
        )
        
    def _build_graph(self):
        """构建Agent的流程图"""
        # 创建图构建器
        builder = StateGraph(State)
        
        # 添加节点
        builder.add_node("query_rewriter", self._query_rewriter_node)
        builder.add_node("retriever", self._retriever_node)
        builder.add_node("generator", self._generator_node)
        builder.add_node("memory_manager", self._memory_manager_node)
        
        # 设置流程
        builder.set_entry_point("query_rewriter")
        builder.add_edge("query_rewriter", "retriever")
        builder.add_edge("retriever", "generator")
        builder.add_edge("generator", "memory_manager")
        builder.add_edge("memory_manager", END)
        
        # 编译图
        memory = SqliteSaver(self.connection)
        self.graph = builder.compile(checkpointer=memory)
    
    def _query_rewriter_node(self, state: State) -> Dict[str, Any]:
        """查询重写节点"""
        # 获取状态里的消息列表
        messages = state["messages"]
        last_user_msg = messages[-1].content
        history = messages[:-1]
        
        # 构建历史文本
        history_text = "\n".join([f"{m.type}: {m.content}" for m in history])

        # 系统prompt
        system_prompt = """你是一个查询改写工具。
        任务：结合历史，将用户的最新输入改写为独立、完整的搜索语句。
        规则：
        1. 补全主语和指代词（如"它"->具体名词）。
        2. 严禁回答问题。
        3. 如果无需改写，原样返回。
        """
        
        # 构建链式调用
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "历史:\n{history}\n\n当前输入: {question}\n\n改写结果:")
        ])
        chain = prompt | self.llm | StrOutputParser()
        rewritten_query = chain.invoke({"history": history_text, "question": last_user_msg})
        
        if self.config.verbose:
            print(f"\n[1.改写] '{last_user_msg}' -> '{rewritten_query}'")
        
        return {"search_query": rewritten_query, "retrieved_context": ""}
    
    def _retriever_node(self, state: State) -> Dict[str, Any]:
        """检索节点"""
        query = state["search_query"]
        
        if len(query) < 2:
            return {"retrieved_context": ""}
        
        if self.config.verbose:
            print(f"[2.检索] 搜索: {query}")
        
        results = self.vector_store.similarity_search(query, k=2)
        
        if not results:
            return {"retrieved_context": "无相关记录"}
        
        # 格式化检索结果
        context_parts = []
        for i, doc in enumerate(results):
            summary = doc.page_content
            raw_quote = doc.metadata.get("raw_content", "无原文")
            if len(raw_quote) > 100:
                raw_quote = raw_quote[:100] + "..."
            
            entry = f"记录{i+1}: {summary}\n   (来源参考: \"{raw_quote}\")"
            context_parts.append(entry)
        
        final_context = "\n\n".join(context_parts)
        
        if self.config.verbose:
            print(f"[2.命中] \n{final_context}")
        
        return {"retrieved_context": final_context}
    
    def _generator_node(self, state: State) -> Dict[str, Any]:
        """生成回复节点"""
        messages = state["messages"]
        context = state["retrieved_context"]
        
        system_tmpl = """你是一个助手。请基于【长期记忆】和【当前对话】回答。
        
        【长期记忆】:
        {context}
        
        注意：
        1. 长期记忆中的信息优先于你的通用知识。
        2. 如果用户询问具体原话，请参考括号中的"来源参考"。
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_tmpl),
            ("placeholder", "{messages}"),
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({"context": context, "messages": messages})
        
        return {"messages": [response]}
    
    def _memory_manager_node(self, state: State) -> Dict[str, Any]:
        """记忆管理节点"""
        messages = state["messages"]
        keep_count = self.config.short_term_window_size
        total_count = len(messages)
        
        if total_count <= keep_count:
            return {}
        
        excess_count = total_count - keep_count
        
        # 强制归档数为偶数（保证不切断对话对）
        if excess_count % 2 != 0:
            excess_count -= 1 
            
        if excess_count <= 0:
            return {}
            
        # 提取要归档的消息
        msgs_to_archive = messages[:excess_count]
        
        # 准备原文
        raw_text = "\n".join([f"{m.type}: {m.content}" for m in msgs_to_archive])
        
        # 生成摘要
        summary_prompt = f"""将以下对话转化为 1 句独立的陈述句事实。
        规则：
        1. 将"我"替换为"User"。
        2. 去除寒暄，只留干货。
        3. 字数控制在 60 字以内。
        
        对话：
        {raw_text}
        """
        
        new_summary = self.llm.invoke(summary_prompt).content.strip()
        
        if self.config.verbose:
            print(f"\n[4.检测] 准备入库: {new_summary}")
        
        # 重复检测
        existing_docs = self.vector_store.similarity_search_with_score(new_summary, k=1)
        
        is_duplicate = False
        if existing_docs:
            doc, score = existing_docs[0]
            if score < 0.35:  # Chroma的score是L2距离
                if self.config.verbose:
                    print(f"   -> 发现重复 (Distance={score:.4f}): '{doc.page_content}'")
                    print("   -> 跳过写入。")
                is_duplicate = True
                
        # 不重复则写入向量库
        if not is_duplicate:
            if self.config.verbose:
                print("   -> 写入向量库。")
           
            self.vector_store.add_texts(
                texts=[new_summary],
                metadatas=[{"raw_content": raw_text, "timestamp": str(uuid.uuid4())}] 
            )
        
        # 返回删除操作
        delete_ops = [RemoveMessage(id=m.id) for m in msgs_to_archive]
        return {"messages": delete_ops}
    
    def chat(self, message: str, thread_id: str = None) -> str:
        """
        与Agent进行对话
        
        Args:
            message: 用户输入的消息
            thread_id: 对话线程ID，用于区分不同对话
            
        Returns:
            Agent的回复
        """
        # 如果没有提供thread_id，则生成一个新的
        if thread_id is None:
            thread_id = str(uuid.uuid4())
            if self.config.verbose:
                print(f"创建新对话线程: {thread_id}")
        
        if self.config.verbose:
            print(f"\n>>> 用户: {message}")
        
        # 配置检查点
        config = {"configurable": {"thread_id": thread_id}}
        
        # 运行Agent图
        for event in self.graph.stream({"messages": [HumanMessage(content=message)]}, config):
            pass
        
        # 获取最终状态
        state = self.graph.get_state(config).values
        if state and state["messages"]:
            last_msg = state["messages"][-1]
            if isinstance(last_msg, AIMessage):
                response = last_msg.content
                if self.config.verbose:
                    print(f"AI: {response}")
                return response
        
        return ""
    
    def get_conversation_history(self, thread_id: str) -> List[BaseMessage]:
        """
        获取指定对话的历史记录
        
        Args:
            thread_id: 对话线程ID
            
        Returns:
            消息历史列表
        """
        config = {"configurable": {"thread_id": thread_id}}
        state = self.graph.get_state(config)
        if state and state.values:
            return state.values.get("messages", [])
        return []
    
    def reset_conversation(self, thread_id: str):
        """
        重置指定对话的历史记录
        
        Args:
            thread_id: 要重置的对话线程ID
        """
        config = {"configurable": {"thread_id": thread_id}}
        self.graph.update_state(config, {"messages": []})
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        获取记忆系统的统计信息
        
        Returns:
            包含记忆系统统计信息的字典
        """
        # 获取向量数据库中的文档数量
        collection = self.vector_store._collection
        count = collection.count() if collection else 0
        
        return {
            "long_term_memories": count,
            "short_term_window_size": self.config.short_term_window_size,
            "embedding_model": self.config.embedding_model,
            "vector_store_path": self.config.vector_store_path
        }
    
    def close(self):
        """关闭Agent，释放资源"""
        if hasattr(self, 'connection'):
            self.connection.close()


# 使用示例
if __name__ == "__main__":
    # 创建Agent配置
    config = MemoryAgentConfig(
        verbose=True,  # 显示详细日志
        checkpoints_db="memory_agent_checkpoints.db"
    )
    
    # 初始化Agent
    agent = MemoryAgent(config)
    
    try:
        # 示例1: 创建新对话
        print("=== 示例1: 新对话 ===")
        response = agent.chat("你好，我叫王五。")
        print(f"AI回复: {response}")
        
        # 示例2: 继续对话
        print("\n=== 示例2: 继续对话 ===")
        response = agent.chat("我是 Python 程序员。")
        print(f"AI回复: {response}")
        
        # 示例3: 测试记忆功能
        print("\n=== 示例3: 测试记忆功能 ===")
        response = agent.chat("我刚才说我叫什么名字？")
        print(f"AI回复: {response}")
        
        # 示例4: 使用特定thread_id
        print("\n=== 示例4: 使用特定thread_id ===")
        thread_id = "test_thread_001"
        response = agent.chat("我们来谈谈机器学习", thread_id=thread_id)
        print(f"AI回复: {response}")
        
        # 获取对话历史
        print("\n=== 获取对话历史 ===")
        history = agent.get_conversation_history(thread_id)
        print(f"对话历史长度: {len(history)}")
        
        # 获取记忆统计
        print("\n=== 记忆系统统计 ===")
        stats = agent.get_memory_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")
            
    finally:
        # 关闭Agent
        agent.close()