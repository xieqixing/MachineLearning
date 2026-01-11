#!/usr/bin/env python3
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages



# 定义状态节点，包含短期记忆消息列表、重写的搜索查询， 实体三元组关系，向量上下文，图谱上下文，融合后的最终上下文，以及待归档消息列表
class State(TypedDict):
    messages: Annotated[list, add_messages]
    search_query: str
    entities: List[str]
    vector_context: str
    graph_context: str
    final_context: str
    msgs_to_archive: List[BaseMessage]