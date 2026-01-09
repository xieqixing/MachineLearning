#!/usr/bin/env python3
import uuid
from typing import Dict, Any, Optional, List
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from .config import MemoryAgentConfig
from .state import State
from .nodes import (
    QueryRewriterNode, 
    RetrieverNode, 
    GeneratorNode, 
    MemoryManagerNode
)
from .utils import (
    setup_environment,
    create_embeddings,
    create_vector_store,
    create_llm,
    create_sqlite_connection
)


class MemoryAgent:
    """具有长期记忆功能的对话Agent"""
    
    def __init__(self, config: Optional[MemoryAgentConfig] = None):
        """
        初始化MemoryAgent
        
        Args:
            config: Agent配置,如为None则使用默认配置
        """
        # 设置环境
        setup_environment()
        
        # 配置
        self.config = config or MemoryAgentConfig()
        
        # 初始化组件
        self._init_components()
        
        # 初始化节点
        self._init_nodes()
        
        # 构建图
        self._build_graph()
    
    def _init_components(self):
        """初始化所有组件"""
        # 创建词向量嵌入模型
        self.embeddings = create_embeddings(self.config.embedding_model)
        
        # 创建向量数据库
        self.vector_store = create_vector_store(
            embedding_function=self.embeddings,
            persist_directory=self.config.vector_store_path
        )
        
        # 创建LLM
        self.llm = create_llm(
            model=self.config.llm_model,
            temperature=self.config.llm_temperature,
            api_key=self.config.llm_api_key,
            api_base=self.config.llm_api_base
        )
        
        # 创建数据库连接
        self.connection = create_sqlite_connection(self.config.checkpoints_db)
    
    def _init_nodes(self):
        """初始化所有节点"""
        # 查询重写节点
        self.query_rewriter_node = QueryRewriterNode(
            llm=self.llm,
            verbose=self.config.verbose
        )
        
        # 检索节点
        self.retriever_node = RetrieverNode(
            vector_store=self.vector_store,
            verbose=self.config.verbose
        )
        
        # 生成回复节点
        self.generator_node = GeneratorNode(
            llm=self.llm,
            verbose=self.config.verbose
        )
        
        # 记忆管理节点
        self.memory_manager_node = MemoryManagerNode(
            llm=self.llm,
            vector_store=self.vector_store,
            short_term_window_size=self.config.short_term_window_size,
            verbose=self.config.verbose
        )
    
    def _build_graph(self):
        """构建Agent的流程图"""
        # 创建图构建器
        builder = StateGraph(State)
        
        # 添加节点
        builder.add_node("query_rewriter", self.query_rewriter_node)
        builder.add_node("retriever", self.retriever_node)
        builder.add_node("generator", self.generator_node)
        builder.add_node("memory_manager", self.memory_manager_node)
        
        # 设置流程
        builder.set_entry_point("query_rewriter")
        builder.add_edge("query_rewriter", "retriever")
        builder.add_edge("retriever", "generator")
        builder.add_edge("generator", "memory_manager")
        builder.add_edge("memory_manager", END)
        
        # 编译图
        memory = SqliteSaver(self.connection)
        self.graph = builder.compile(checkpointer=memory)
    
    def chat(self, message: str, thread_id: str = None) -> str:
        """
        与Agent进行对话
        
        Args:
            message: 用户输入的消息
            thread_id: 对话线程ID,用于区分不同对话
            
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
        for event in self.graph.stream(
            {"messages": [HumanMessage(content=message)]}, 
            config
        ):
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
    
    def clear_long_term_memory(self):
        """清空长期记忆"""
        self.vector_store.delete_collection()
        # 重新创建集合
        self.vector_store = create_vector_store(
            embedding_function=self.embeddings,
            persist_directory=self.config.vector_store_path
        )
        if self.config.verbose:
            print("长期记忆已清空")
    
    def close(self):
        """关闭Agent,释放资源"""
        if hasattr(self, 'connection'):
            self.connection.close()
            if self.config.verbose:
                print("Agent已关闭")