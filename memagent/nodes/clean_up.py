from typing import Dict, Any
from langchain_core.messages import RemoveMessage


# 清理归档消息节点
class CleanUpNode:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        msgs = state.get("msgs_to_archive", [])
        if not msgs: return {}
        
        return {"messages": [RemoveMessage(id=m.id) for m in msgs]}
