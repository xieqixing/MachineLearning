#!/usr/bin/env python3
from typing import Dict, Any
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableConfig


class VectorRetrieverNode:
    """检索节点"""
    
    def __init__(self, vector_store: Chroma, verbose: bool = True, max_distance: float = 0.35):
        self.vector_store = vector_store
        self.verbose = verbose
        self.max_distance = max_distance
    
    # 完成检索逻辑
    def __call__(self, state: Dict[str, Any], config: RunnableConfig = None) -> Dict[str, Any]:
        # 查询配置
        config = config or {}
        conf = config.get("configurable", {})

        # 消融实验开关
        if not conf.get("use_vector_memory", True):
            return {"vector_context": ""}
        
        # 获取搜索查询
        query = state["search_query"]
        
        # 如果查询过短，直接返回空结果
        if len(query) < 2:
            return {"vector_context": ""}
        
        if self.verbose:
            print(f"[2.检索] 搜索: {query}")
        
        # 执行相似度搜索
        results = self.vector_store.similarity_search(query, k=2)
        
        # 如果没有结果，返回默认提示
        if not results:
            return {"vector_context": "无相关记录"}

        # # 过滤：distance 越小越相似；太大视为不相关
        # kept = [(doc, dist) for doc, dist in results if dist <= self.max_distance]

        # if self.verbose:
        #     print("[2.检索] distances:", [round(dist, 4) for _, dist in results])
        #     print("[2.检索] kept:", [round(dist, 4) for _, dist in kept])

        # if not kept:
        #     return {"vector_context": ""}
        
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
        
        return {"vector_context": final_context}