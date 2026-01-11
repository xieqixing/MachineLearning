#!/usr/bin/env python3
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sqlite3
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from .graphDB import EnhancedGraphDB

# 设置环境变量
def setup_environment():
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


# 创建词向量嵌入模型
def create_embeddings(model_name: str = "all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=model_name)


# 创建向量数据库
def create_vector_store(
    embedding_function: HuggingFaceEmbeddings,
    persist_directory: str = "./chroma_db",
    collection_name: str = "long_term_memory"
) -> Chroma:
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory=persist_directory
    )

# 创建LLM实例
def create_llm(
    model: str = "qwen-plus",
    temperature: float = 0.1,
    api_key: str = None,
    api_base: str = None
) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        openai_api_key=api_key,
        openai_api_base=api_base,
    )


# 创建SQLite数据库连接
def create_sqlite_connection(db_path: str = "checkpoints.db"):
    return sqlite3.connect(db_path, check_same_thread=False)


# 创建图数据库连接
def create_graph_database():
    return EnhancedGraphDB()
        
    