from typing import Dict, Any


# 回复生成节点
class ContextFusionNode:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        v_ctx = state.get("vector_context", "")
        g_ctx = state.get("graph_context", "")
        
        final = ""
        if v_ctx:
            final += f"【非结构化历史记录】:\n{v_ctx}\n\n"
        if g_ctx:
            final += f"【结构化知识图谱】:\n{g_ctx}\n\n"
            
        if not final:
            final = "无历史记忆。"
            
        return {"final_context": final}