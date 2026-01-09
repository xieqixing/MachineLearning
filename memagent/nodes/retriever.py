#!/usr/bin/env python3
from typing import Dict, Any
from langchain_chroma import Chroma


class RetrieverNode:
    """检索节点"""
    
    def __init__(self, vector_store: Chroma, verbose: bool = True):
        self.vector_store = vector_store
        self.verbose = verbose
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """执行检索"""
        # 获取搜索查询
        query = state["search_query"]
        
        # 如果查询过短，直接返回空结果
        if len(query) < 2:
            return {"retrieved_context": ""}
        
        if self.verbose:
            print(f"[2.检索] 搜索: {query}")
        
        # 执行相似度搜索
        results = self.vector_store.similarity_search(query, k=2)
        
        # 如果没有结果，返回默认提示
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
        
        # 拼接最终上下文
        final_context = "\n\n".join(context_parts)
        
        if self.verbose:
            print(f"[2.命中] \n{final_context}")
        
        return {"retrieved_context": final_context}