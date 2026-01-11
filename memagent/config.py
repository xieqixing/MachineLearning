#!/usr/bin/env python3
from dataclasses import dataclass
from typing import Optional


@dataclass
class MemoryAgentConfig:
    """Agent配置类"""
    llm_model: str = "qwen-plus"
    llm_temperature: float = 0.1
    llm_api_key: str = "sk-0a3f574aeed045e3b4d2584e5bc7c291"
    llm_api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embedding_model: str = "all-MiniLM-L6-v2"
    short_term_window_size: int = 4
    summary_batch_size: int = 2
    vector_store_path: str = "./chroma_db"
    checkpoints_db: str = "checkpoints.db"
    verbose: bool = True
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'MemoryAgentConfig':
        """从字典创建配置"""
        return cls(**{
            k: v for k, v in config_dict.items() 
            if k in cls.__dataclass_fields__
        })