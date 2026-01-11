
from typing import Dict, Any
from ..graphDB import EnhancedGraphDB
from langchain_core.runnables import RunnableConfig


# 图数据库检索节点
class GraphRetrieverNode:
    def __init__(self, graphDB: EnhancedGraphDB, verbose: bool = True):
        self.graphDB = graphDB
        self.verbose = verbose

    def __call__(self, state: Dict[str, Any], config: RunnableConfig = None) -> Dict[str, Any]:
        config = config or {}
        conf = config.get("configurable", {})
        
        # 消融实验开关
        if not conf.get("use_graph_memory", True):
            return {"graph_context": ""}
        
        # 获取实体列表，如果没有实体，直接返回空结果
        entities = state["entities"]
        if not entities and self.verbose:
            print(f"[2.图谱] 没有实体。")
            return {"graph_context": ""}
        
        # 查询图数据库，获取相关关系
        context = self.graphDB.get_context(entities, depth=1)

        # 输出检索结果，便于调试
        if self.verbose:
            if context:
                print(f"[2.图谱] 命中关系:\n{context}")
            else:
                print(f"[2.图谱] 未命中任何关系。")

        return {"graph_context": context}