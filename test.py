#!/usr/bin/env python3
import os
import uuid
import sqlite3
from typing import TypedDict, Annotated, List
from operator import itemgetter

# LangChain / LangGraph imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, RemoveMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages

# ================= 配置区域 =================
# 为了演示效果，我们将短期记忆窗口设置得很小，以便快速触发“归档”动作
SHORT_TERM_WINDOW_SIZE = 4  # 保留最近的4条消息
SUMMARY_BATCH_SIZE = 2      # 每次超限时，归档最早的2条消息

# 使用 HuggingFace 的轻量级模型进行 Embeddings（免费，本地运行）
# 第一次运行会自动下载模型 (约 80MB)
EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 初始化本地向量数据库
VECTOR_STORE = Chroma(
    collection_name="long_term_memory",
    embedding_function=EMBEDDING_MODEL,
    persist_directory="./chroma_db"  # 数据持久化到本地文件夹
)

# LLM 配置 (使用你的配置)
llm = ChatOpenAI(
    model="qwen-plus",
    temperature=0.1,
    # 建议将 Key 放入环境变量，这里为了演示直接写入（请替换为你自己的 Key）
    openai_api_key="sk-2770a3f619c14f31a87d47924de34af2", 
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# ================= 1. 状态定义 =================
class State(TypedDict):
    # 消息列表（短期记忆）
    messages: Annotated[list, add_messages]
    # 改写后的独立查询语句（用于检索）
    search_query: str
    # 从向量库检索到的上下文
    retrieved_context: str

# ================= 2. 核心功能函数 =================

def archive_logic(messages: List[BaseMessage]):
    """
    将旧消息总结并存入向量库
    """
    if len(messages) < SUMMARY_BATCH_SIZE:
        return
    
    # 1. 提取要归档的消息 (最早的几条)
    msgs_to_archive = messages[:SUMMARY_BATCH_SIZE]
    
    # 2. 生成摘要 (让 LLM 总结这几条对话的内容)
    # 我们把原始对话转成字符串
    chat_text = "\n".join([f"{m.type}: {m.content}" for m in msgs_to_archive])
    prompt = f"请简要总结以下对话的内容，保留关键信息（如人名、事件、偏好）：\n\n{chat_text}"
    summary = llm.invoke(prompt).content
    
    print(f"\n[系统] 正在归档旧记忆 -> 摘要: {summary}")
    
    # 3. 存入向量数据库
    # 我们存储摘要，元数据里可以放原始对话，方便检索
    VECTOR_STORE.add_texts(
        texts=[f"历史对话摘要: {summary}. 原始详情: {chat_text}"],
        metadatas=[{"source": "conversation_archive"}]
    )

# ================= 3. 节点定义 =================

def memory_manager_node(state: State):
    """
    节点：管理短期记忆。
    如果消息太多，就归档旧消息到向量库，并从 State 中删除它们。
    """
    messages = state["messages"]
    
    # 如果消息数超过窗口限制
    if len(messages) > SHORT_TERM_WINDOW_SIZE:
        # 1. 触发归档逻辑（存入向量库）
        archive_logic(messages)
        
        # 2. 构建“删除消息”的操作
        # LangGraph 中使用 RemoveMessage(id) 来删除特定消息
        delete_ops = [RemoveMessage(id=m.id) for m in messages[:SUMMARY_BATCH_SIZE]]
        
        return {"messages": delete_ops}
    
    return {} # 状态无变化

def query_rewriter_node(state: State):
    """
    节点：查询重写。
    结合短期记忆，把用户的最后一句话改写成独立的搜索查询。
    """
    messages = state["messages"]
    last_user_msg = messages[-1].content
    
    # 如果只是第一句话，或者历史很短，可能不需要改写，但为了统一流程，我们还是做一下
    if len(messages) <= 1:
        return {"search_query": last_user_msg}

    # Prompt: 让 LLM 结合上下文澄清问题
    system_prompt = """你是一个专业的查询改写助手。
    你的任务是将用户最新的问题改写成一个独立的、语义完整的搜索查询。
    用户的问题可能包含指代词（如“它”、“那个”），你需要根据聊天历史将其补全。
    只返回改写后的查询，不要解释。"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{question}"),
    ])
    
    # 排除最后一条消息作为历史，最后一条是当前问题
    history = messages[:-1]
    chain = prompt | llm | StrOutputParser()
    
    rewritten_query = chain.invoke({"chat_history": history, "question": last_user_msg})
    print(f"\n[系统] 原始问题: {last_user_msg} -> 改写后检索词: {rewritten_query}")
    
    return {"search_query": rewritten_query}

def retriever_node(state: State):
    """
    节点：检索。
    使用改写后的查询去向量库搜索长期记忆。
    """
    query = state["search_query"]
    
    # 从向量库检索最相关的 2 条片段
    docs = VECTOR_STORE.similarity_search(query, k=2)
    
    context_text = "\n\n".join([d.page_content for d in docs])
    
    if context_text:
        print(f"[系统] 检索到长期记忆: {context_text[:50]}...")
    else:
        print("[系统] 未检索到相关长期记忆。")
        context_text = "无相关历史记录。"
        
    return {"retrieved_context": context_text}

def generator_node(state: State):
    """
    节点：生成回复。
    综合 系统提示 + 长期记忆(RAG) + 短期记忆 生成最终答案。
    """
    messages = state["messages"]
    context = state["retrieved_context"]
    
    # 构建 Prompt
    system_prompt = (
        "你是一个乐于助人的 AI 助手。\n"
        "请利用以下检索到的【长期记忆】和当前的【对话历史】来回答用户。\n"
        "如果长期记忆中有相关信息，请优先参考。\n"
        "--------------------\n"
        "【长期记忆】:\n"
        "{context}\n"
        "--------------------"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{messages}"), # 这里的 messages 是 LangGraph 自动管理的短期记忆
    ])
    
    chain = prompt | llm
    
    # 注意：我们将 context 注入到 prompt 模板中，messages 由 graph 传递
    response = chain.invoke({"context": context, "messages": messages})
    
    return {"messages": [response]}

# ================= 4. 建图 =================

conn = sqlite3.connect("checkpoints_advanced.db", check_same_thread=False)
memory = SqliteSaver(conn)

builder = StateGraph(State)

# 添加节点
builder.add_node("memory_manager", memory_manager_node)
builder.add_node("query_rewriter", query_rewriter_node)
builder.add_node("retriever", retriever_node)
builder.add_node("generator", generator_node)

# 定义边（执行流程）
# 1. 入口 -> 检查记忆是否需要归档
builder.set_entry_point("memory_manager")

# 2. 记忆管理 -> 查询改写
builder.add_edge("memory_manager", "query_rewriter")

# 3. 查询改写 -> 检索
builder.add_edge("query_rewriter", "retriever")

# 4. 检索 -> 生成回复
builder.add_edge("retriever", "generator")

# 5. 生成回复 -> 结束
builder.add_edge("generator", END)

graph = builder.compile(checkpointer=memory)

# ================= 5. 模拟运行 =================

def run_chat(message_text, thread_id):
    """辅助运行函数"""
    config = {"configurable": {"thread_id": thread_id}}
    print(f"\n--- 用户: {message_text} ---")
    
    # 发送用户消息
    input_state = {"messages": [HumanMessage(content=message_text)]}
    
    for event in graph.stream(input_state, config):
        pass # 我们在节点内部打印了日志，这里主要为了驱动流程运行
        
    # 获取最后一次状态中的回复
    snapshot = graph.get_state(config)
    if snapshot.values and snapshot.values["messages"]:
        last_msg = snapshot.values["messages"][-1]
        if isinstance(last_msg, AIMessage):
            print(f"AI: {last_msg.content}")

# 初始化一个会话 ID
tid = str(uuid.uuid4())

print(f"当前会话 ID: {tid}")
print(f"短期记忆窗口: {SHORT_TERM_WINDOW_SIZE}, 超限归档数: {SUMMARY_BATCH_SIZE}")

# --- 第一阶段：填充记忆 ---
# 我们连续发送几条消息，让 AI 记住信息，并触发记忆归档
run_chat("你好，我叫小明。", tid)
run_chat("我是一名软件工程师。", tid)
run_chat("我最喜欢的编程语言是 Python。", tid)
# 此时应该有 6 条消息 (3 User + 3 AI)，超过窗口 4
# 下一次对话开始时，`memory_manager` 会检测到并归档最早的 2 轮对话

# --- 第二阶段：触发归档与 RAG ---
print("\n>>> 下一条消息将触发【归档】和【检索】 <<<")
# 这里的“它”指代模糊，Query Rewriter 应该能将其改写为“小明最喜欢的编程语言是什么”
# 且因为原始消息可能已被归档，Generator 必须依靠 Retriever 从向量库找回信息
run_chat("它有什么优点？（测试模糊提问）", tid) 

# --- 第三阶段：测试完全遗忘后的回忆 ---
print("\n>>> 模拟很久之后，再次询问个人信息 <<<")
# 此时短期记忆里可能已经没有“我叫小明”这条原始消息了（被删除了）
# AI 必须通过 RAG 从 Chroma 数据库中找到之前归档的摘要
run_chat("还记得我叫什么名字吗？", tid)

print("\n✅ 演示结束。数据已存入 chroma_db 和 sqlite。")