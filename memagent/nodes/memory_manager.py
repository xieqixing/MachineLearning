#!/usr/bin/env python3
import uuid
from typing import Dict, Any
from langchain_core.messages import RemoveMessage
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma


class MemoryManagerNode:
    """记忆管理节点"""
    
    def __init__(self, llm: ChatOpenAI, vector_store: Chroma, 
                 short_term_window_size: int = 4, verbose: bool = True):
        self.llm = llm
        self.vector_store = vector_store
        self.short_term_window_size = short_term_window_size
        self.verbose = verbose
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """管理记忆：归档短期记忆到长期记忆"""
        # 获取当前短期记忆消息列表
        messages = state["messages"]
        keep_count = self.short_term_window_size
        total_count = len(messages)
        
        # 如果消息数未超出窗口大小，直接返回空操作
        if total_count <= keep_count:
            return {}
        
        excess_count = total_count - keep_count
        
        # 强制归档数为偶数（保证不切断对话对）
        if excess_count % 2 != 0:
            excess_count -= 1 
            
        if excess_count <= 0:
            return {}
            
        # 切片提取要归档的消息
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
        
        if self.verbose:
            print(f"\n[4.检测] 准备入库: {new_summary}")
        
        # 重复检测
        existing_docs = self.vector_store.similarity_search_with_score(
            new_summary, 
            k=1
        )
        
        is_duplicate = False
        if existing_docs:
            doc, score = existing_docs[0]
            if score < 0.35:  # Chroma的score是L2距离
                if self.verbose:
                    print(f"   -> 发现重复 (Distance={score:.4f}): '{doc.page_content}'")
                    print("   -> 跳过写入。")
                is_duplicate = True
                
        # 不重复则写入向量库
        if not is_duplicate:
            if self.verbose:
                print("   -> 写入向量库。")
           
            self.vector_store.add_texts(
                texts=[new_summary],
                metadatas=[{
                    "raw_content": raw_text, 
                    "timestamp": str(uuid.uuid4())
                }] 
            )
        
        # 返回删除操作
        delete_ops = [RemoveMessage(id=m.id) for m in msgs_to_archive]
        return {"messages": delete_ops}