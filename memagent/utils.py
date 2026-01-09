#!/usr/bin/env python3
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sqlite3
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from .graphDB import EnhancedGraphDB


def setup_environment():
    """设置环境变量"""
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


def create_embeddings(model_name: str = "all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    """创建词向量嵌入模型"""
    return HuggingFaceEmbeddings(model_name=model_name)


def create_vector_store(
    embedding_function: HuggingFaceEmbeddings,
    persist_directory: str = "./chroma_db",
    collection_name: str = "long_term_memory"
) -> Chroma:
    """创建向量数据库"""
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory=persist_directory
    )


def create_llm(
    model: str = "qwen-plus",
    temperature: float = 0.1,
    api_key: str = None,
    api_base: str = None
) -> ChatOpenAI:
    """创建LLM实例"""
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        openai_api_key=api_key,
        openai_api_base=api_base,
    )


def create_sqlite_connection(db_path: str = "checkpoints.db"):
    """创建SQLite数据库连接"""
    return sqlite3.connect(db_path, check_same_thread=False)



def create_graph_database():
    """创建图数据库连接"""
    return EnhancedGraphDB()
        
    