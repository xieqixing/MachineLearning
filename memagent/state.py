#!/usr/bin/env python3
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


class State(TypedDict):
    """Agent状态定义"""
    messages: Annotated[list, add_messages]
    search_query: str
    retrieved_context: str