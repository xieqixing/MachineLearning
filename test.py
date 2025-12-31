#!/usr/bin/env python3
"""
LangGraph 最小可运行示例（免费 LLM + SQLite 检查点）
运行前：
    export OPENAI_API_KEY="gsk_你的GroqKey"
    export OPENAI_BASE_URL="https://api.groq.com/openai/v1"
"""

import os
import uuid
import sqlite3
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages

# 0. 模型
llm = ChatOpenAI(
    model="qwen-plus",
    temperature=0,
    openai_api_key="sk-2770a3f619c14f31a87d47924de34af2",
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# ----------------  1. 状态定义  ----------------
class State(TypedDict):
    messages: Annotated[list, add_messages]

# ----------------  2. 节点函数  ----------------
def chat_node(state: State):
    """调用 LLM 并返回消息"""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# ----------------  3. 建图 + 检查点  ----------------
conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
memory = SqliteSaver(conn)

builder = StateGraph(State)
builder.add_node("chat", chat_node)
builder.set_entry_point("chat")
builder.add_edge("chat", END)
graph = builder.compile(checkpointer=memory)

# ----------------  4. 第一次对话  ----------------
thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": thread_id}}

print("=== 第 1 轮 ===")
init_msg = [("user", "我叫小明，请记住")]
for event in graph.stream({"messages": init_msg}, config):
    ai_reply = event["chat"]["messages"][-1].content
    print(f"AI: {ai_reply}")

# ----------------  5. 模拟重启后对话（断点续跑） ----------------
print("\n=== 重启后 ===")
new_msg = [("user", "我叫什么？")]
for event in graph.stream({"messages": new_msg}, config):
    ai_reply = event["chat"]["messages"][-1].content
    print(f"AI: {ai_reply}")

print("\n✅ 检查点已落盘，可反复续跑；如需清空记忆请删除 checkpoints.db")