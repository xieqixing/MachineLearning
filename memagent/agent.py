#!/usr/bin/env python3
import uuid
from typing import Dict, Any, Optional, List
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from .config import MemoryAgentConfig
from .state import State
from .nodes import *
from .utils import (
    setup_environment,
    create_embeddings,
    create_vector_store,
    create_llm,
    create_sqlite_connection,
    create_graph_database
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

        # 创建图数据库连接
        self.graph_db = create_graph_database()
    
    # 初始化所有节点
    def _init_nodes(self):
        # 清理归档消息节点
        self.clean_up_node = CleanUpNode(
            verbose=self.config.verbose
        )

        # 回复生成节点
        self.context_fusion = ContextFusionNode(
            verbose=self.config.verbose
        )

        # 生成回复节点
        self.generator_node = GeneratorNode(
            llm=self.llm,
            verbose=self.config.verbose
        )

        # 存入图数据库节点
        self.graph_archiver_node = GraphArchiverNode(
            llm=self.llm,
            graphDB=self.graph_db,
            verbose=self.config.verbose
        )

        # 查询图数据库节点
        self.graph_retriever_node = GraphRetrieverNode(
            graphDB=self.graph_db,
            verbose=self.config.verbose
        )

        # 内存管理路由
        self.memory_router_node = MemoryRouterNode(
            short_term_window_size=self.config.short_term_window_size,
            verbose=self.config.verbose
        )

        # 查询重写节点
        self.query_rewriter_node = QueryRewriterNode(
            llm=self.llm,
            verbose=self.config.verbose
        )

        # 存入向量数据库节点
        self.vector_archiver_node = VectorArchiverNode(
            llm=self.llm,
            vector_store=self.vector_store,
            verbose=self.config.verbose
        )
        
        # 查询向量数据库节点
        self.vector_retriever_node = VectorRetrieverNode(
            vector_store=self.vector_store,
            verbose=self.config.verbose
        )
        

    # 构建agent流程图
    def _build_graph(self):
        # 构建agent的流程图
        workflow = StateGraph(State)

        # 1. 添加节点
        workflow.add_node("query_processing", self.query_rewriter_node)
        workflow.add_node("vector_retriever", self.vector_retriever_node)
        workflow.add_node("graph_retriever", self.graph_retriever_node)
        workflow.add_node("context_fusion", self.context_fusion)
        workflow.add_node("generator", self.generator_node)
        workflow.add_node("memory_router", self.memory_router_node)
        workflow.add_node("vector_archiver", self.vector_archiver_node)
        workflow.add_node("graph_archiver", self.graph_archiver_node)
        workflow.add_node("cleanup", self.clean_up_node)

        # 2. 定义边
        workflow.set_entry_point("query_processing")
        workflow.add_edge("query_processing", "vector_retriever")
        workflow.add_edge("vector_retriever", "graph_retriever")
        workflow.add_edge("graph_retriever", "context_fusion")
        workflow.add_edge("context_fusion", "generator")
        workflow.add_edge("generator", "memory_router")

        # 条件边
        def should_archive(state: State):
            if state["msgs_to_archive"]:
                return "archive"
            return "end"

        workflow.add_conditional_edges(
            "memory_router",
            should_archive,
            {
                "archive": "vector_archiver",
                "end": END
            }
        )

        workflow.add_edge("vector_archiver", "graph_archiver")
        workflow.add_edge("graph_archiver", "cleanup")
        workflow.add_edge("cleanup", END)
        
        # 编译图
        memory = SqliteSaver(self.connection)
        self.graph = workflow.compile(checkpointer=memory)
    
    def chat(self, message: str, thread_id: str = None, enable_vector=True, enable_graph=True) -> str:
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
        config = {"configurable": {"thread_id": thread_id,
                                   "use_vector_memory": enable_vector,
                                   "use_graph_memory": enable_graph}}
        
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