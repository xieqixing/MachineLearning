#!/usr/bin/env python3
import uuid
from typing import Dict, Any
from langchain_core.messages import RemoveMessage


# 内存管理路由：监控对话是否超过了短期窗口大小
class MemoryRouterNode:
    
    def __init__(self, short_term_window_size: int = 4, verbose: bool = True):
        self.short_term_window_size = short_term_window_size
        self.verbose = verbose
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        
        # 获取当前短期记忆消息列表
        messages = state["messages"]
        keep_count = self.short_term_window_size
        total_count = len(messages)
        
        # 如果消息数未超出窗口大小，直接返回空操作
        if total_count <= keep_count:
            return {"msgs_to_archive": []}
        
        excess_count = total_count - keep_count
        
        # 强制归档数为偶数（保证不切断对话对）
        if excess_count % 2 != 0:
            excess_count -= 1 
            
        if excess_count <= 0:
            return {"msgs_to_archive": []}
            
        # 切片提取要归档的消息
        msgs_to_archive = messages[:excess_count]

        return {"msgs_to_archive": msgs_to_archive}