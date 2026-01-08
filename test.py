#!/usr/bin/env python3
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 设置 HuggingFace 镜像加速

import uuid
import sqlite3
from typing import TypedDict, Annotated, List
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, RemoveMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages


# 全局配置
SHORT_TERM_WINDOW_SIZE = 4  # 短期记忆窗口大小（消息条数），先设置为4条
SUMMARY_BATCH_SIZE = 2      # 当短期记忆超出4条的时候，一次性归档两条消息（用户和AI各一条）
EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # 采用的词向量嵌入模型

# 初始化本地向量数据库，用于长期记忆存储
VECTOR_STORE = Chroma(
    collection_name="long_term_memory",
    embedding_function=EMBEDDING_MODEL,
    persist_directory="./chroma_db"  # 数据持久化到本地文件夹
)

# LLM api 初始化
llm = ChatOpenAI(
    model="qwen-plus",
    temperature=0.1,
    openai_api_key="sk-2770a3f619c14f31a87d47924de34af2", 
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


# 定义状态节点，包含短期记忆消息列表、重写的搜索查询和检索到的上下文
class State(TypedDict):
    messages: Annotated[list, add_messages]  # 当某个节点返回 {"messages": [新消息]} 时，LangGraph 不会覆盖旧列表，而是自动 append (追加)
    search_query: str
    retrieved_context: str 


# 根据State里面的message（短期记忆）重写问题，最终将重写的问题填到State的search_query字段里
# prompt的格式是：【历史】... 【当前输入】... 【改写结果】: 
def query_rewriter_node(state: State):
    clean_context = "" # 清空上一轮上下文
    
    # 获取状态里的消息列表
    messages = state["messages"]
    last_user_msg = messages[-1].content    # 最新用户输入
    history = messages[:-1]                 # 历史消息（不含最新用户输入）
    
    # 构建历史文本
    history_text = "\n".join([f"{m.type}: {m.content}" for m in history])


    # 系统prompt，指导模型如何改写问题
    system_prompt = """你是一个查询改写工具。
    任务：结合历史，将用户的最新输入改写为独立、完整的搜索语句。
    规则：
    1. 补全主语和指代词（如“它”->具体名词）。
    2. 严禁回答问题。
    3. 如果无需改写，原样返回。
    """
    
    # LangChain 链式调用：Prompt -> LLM -> 解析为字符串
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "历史:\n{history}\n\n当前输入: {question}\n\n改写结果:")
    ])
    chain = prompt | llm | StrOutputParser()
    rewritten_query = chain.invoke({"history": history_text, "question": last_user_msg})
    

    # 输出改写结果，便于调试
    print(f"\n[1.改写] '{last_user_msg}' -> '{rewritten_query}'")
    
    # 返回改写后的查询，并清空retrieved_context
    return {"search_query": rewritten_query, "retrieved_context": clean_context}


# 通过上一个节点重写的问题去向量数据库里面进行检索，找到最相关的两条消息，并将结果填到State的retrieved_context字段里
# 存的格式是：记录1: 摘要 (来源参考: "原文")
def retriever_node(state: State):
    # 获取上个节点的搜索查询
    query = state["search_query"]
    
    # 如果查询过短，直接返回空结果
    if len(query) < 2:
        return {"retrieved_context": ""}

    print(f"[2.检索] 搜索: {query}")
    
    results = VECTOR_STORE.similarity_search(query, k=2)    # 将 query 转成向量，去数据库比对，找出最相似的 top 2 条记录
    
    # 如果没有结果，返回默认提示
    if not results:
        return {"retrieved_context": "无相关记录"}

    
    # 格式：【事实】摘要 (【来源】原文)，这样 LLM 既能快速理解事实，也能在需要时引用原话
    context_parts = []
    for i, doc in enumerate(results):
        summary = doc.page_content
        raw_quote = doc.metadata.get("raw_content", "无原文")
        # 限制原文引用的长度，防止 Context 爆炸
        if len(raw_quote) > 100: raw_quote = raw_quote[:100] + "..."
        
        entry = f"记录{i+1}: {summary}\n   (来源参考: \"{raw_quote}\")"
        context_parts.append(entry)
    
    # 拼接最终上下文
    final_context = "\n\n".join(context_parts)
    print(f"[2.命中] \n{final_context}")
    
    return {"retrieved_context": final_context}


# 生成回复的一个节点，将短期记忆和长期记忆结合起来生成回复，并将回复放入State的messages字段里（通过加方法append）
def generator_node(state: State):
    # 获取短期记忆和搜索到的长期记忆
    messages = state["messages"]
    context = state["retrieved_context"]
    
    # 构建系统提示，指导模型如何利用短期记忆和长期记忆回答问题
    system_tmpl = """你是一个助手。请基于【长期记忆】和【当前对话】回答。
    
    【长期记忆】:
    {context}
    
    注意：
    1. 长期记忆中的信息优先于你的通用知识。
    2. 如果用户询问具体原话，请参考括号中的“来源参考”。
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_tmpl),
        ("placeholder", "{messages}"),
    ])
    
    chain = prompt | llm
    response = chain.invoke({"context": context, "messages": messages})
    
    return {"messages": [response]}


# 归档清理节点，将超过短期记忆窗口的消息进行归档处理，存入向量数据库
# 归档时会生成摘要，并进行重复检测，避免存入重复信息
def memory_manager_node(state: State):
    # 获取当前短期记忆消息列表
    messages = state["messages"]
    
    # 获取设定的短期记忆窗口大小
    keep_count = SHORT_TERM_WINDOW_SIZE
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
    
    # 生成摘要的 prompt
    summary_prompt = f"""将以下对话转化为 1 句独立的陈述句事实。
    规则：
    1. 将“我”替换为“User”。
    2. 去除寒暄，只留干货。
    3. 字数控制在 60 字以内。
    
    对话：
    {raw_text}
    """
    
    new_summary = llm.invoke(summary_prompt).content.strip()
    # new_summary = new_summary[:120] 
    
    print(f"\n[4.检测] 准备入库: {new_summary}")
    
    
    # 重复检测，查找最相似的一条记录，看看是否足够相似
    existing_docs = VECTOR_STORE.similarity_search_with_score(new_summary, k=1)
    
    is_duplicate = False
    if existing_docs:
        doc, score = existing_docs[0]

        # Chroma 的 score 是 L2 距离 (越小越相似)，0.35 约为 Cosine 0.94 左右的相似度
        if score < 0.35:
            print(f"   -> 发现重复 (Distance={score:.4f}): '{doc.page_content}'")
            print("   -> 跳过写入。")
            is_duplicate = True
            
    # 不重复的话，就写入向量库
    if not is_duplicate:
        print("   -> 写入向量库。")
       
        VECTOR_STORE.add_texts(
            texts=[new_summary],
            metadatas=[{"raw_content": raw_text, "timestamp": str(uuid.uuid4())}] 
        )
    
    # 删除操作列表
    delete_ops = [RemoveMessage(id=m.id) for m in msgs_to_archive]
    
    return {"messages": delete_ops}     # 返回删除操作列表，LangGraph 会执行删除


# 构建agent的流程图
builder = StateGraph(State)

builder.add_node("query_rewriter", query_rewriter_node)
builder.add_node("retriever", retriever_node)
builder.add_node("generator", generator_node)
builder.add_node("memory_manager", memory_manager_node)

builder.set_entry_point("query_rewriter")
builder.add_edge("query_rewriter", "retriever")
builder.add_edge("retriever", "generator")
builder.add_edge("generator", "memory_manager")
builder.add_edge("memory_manager", END)

conn = sqlite3.connect("checkpoints_v5.db", check_same_thread=False)
memory = SqliteSaver(conn)
graph = builder.compile(checkpointer=memory)


# 一个简短的验证测试
def run_chat(text, thread_id):
    print(f"\n>>> 用户: {text}")
    config = {"configurable": {"thread_id": thread_id}}
    for event in graph.stream({"messages": [HumanMessage(content=text)]}, config):
        pass
    
    state = graph.get_state(config).values
    if state and state["messages"]:
        last_msg = state["messages"][-1]
        if isinstance(last_msg, AIMessage):
            print(f"AI: {last_msg.content}")



# 测试正式开始

# 生成唯一线程ID
tid = str(uuid.uuid4())
print(f"Thread ID: {tid}")

# 1. 制造数据
run_chat("你好，我叫王五。", tid)
run_chat("我是 Python 程序员。", tid)
run_chat("我特别讨厌 Java。", tid) 
# 此时积累了 6 条消息 (3对)，超过窗口 4
# memory_manager 应触发：
# total=6, keep=4, excess=2. 删除前 2 条 (你好 + 你好回复)。
# 留下：我是Python程序员...

# 2. 测试重复检测
# 如果用户再次强调同样的事
run_chat("一定要记住，我是 Python 程序员。", tid)
# 此时 memory_manager 再次运行，试图归档 "我是Python程序员..." 这一对
# 因为之前应该已经存过类似的摘要，这次应当触发 [发现重复] -> [跳过写入]

# 3. 测试原文回溯
print("\n=== 测试原文引用能力 ===")
# 此时 "我特别讨厌 Java" 这句话应该还在短期记忆或刚被归档
run_chat("我刚才说我叫什么？原话是什么？", tid)
# 预期 AI 能答出 "Java"，并可能引用 metadata 中的原话

print("\n✅ V5.0 测试完成。")