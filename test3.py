#!/usr/bin/env python3
import os
# ================= 配置与工具初始化 =================
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import uuid
import sqlite3
import tiktoken
import networkx as nx
import difflib  # 用于模糊匹配
import json
from typing import TypedDict, Annotated, List, Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, RemoveMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableConfig
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages

# 1. 配置参数
SHORT_TERM_TOKEN_LIMIT = 600  # 调低以便快速触发归档
CHUNK_SIZE = 600              # 归档到向量数据库时的文本块最大大小
CHUNK_OVERLAP = 50            # 切片重叠 50 字符，防止语义在切分处断裂

# embedding模型采用 HuggingFace 的 all-MiniLM-L6-v2
EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 初始化本地向量数据库，用于长期记忆存储
VECTOR_STORE = Chroma(
    collection_name="dual_memory_vector",
    embedding_function=EMBEDDING_MODEL,
    persist_directory="./chroma_dual_db"
)

# LLM api 初始化，采用 Qwen-Plus
llm = ChatOpenAI(
    model="qwen-plus", 
    temperature=0.1,
    openai_api_key="sk-2770a3f619c14f31a87d47924de34af2",
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)  # 文本切分器，用于归档时将长文本切分为更小块
encoding = tiktoken.get_encoding("cl100k_base")     # 用于计算 Token 数量



# 3. 模拟图数据库 (基于 NetworkX)
class EnhancedGraphDB:
    def __init__(self):
        self.graph = nx.DiGraph()
    
    # 添加三元组: [[Head, Relation, Tail], ...]
    def add_triples(self, triples: List[List[str]]):
        for item in triples:
            if len(item) == 3:
                head, relation, tail = item
                # 标准化：去除首尾空格，转小写（可选）
                head = head.strip()
                tail = tail.strip()
                self.graph.add_edge(head, tail, relation=relation)
                print(f"      [Graph Write] ({head}) --[{relation}]--> ({tail})")

    # 模糊匹配找到最相似的节点
    def _fuzzy_match_node(self, query_entity: str) -> str:
        all_nodes = list(self.graph.nodes())
        # 使用 difflib 查找最接近的匹配，cutoff=0.6 表示相似度至少 60%
        matches = difflib.get_close_matches(query_entity, all_nodes, n=1, cutoff=0.6)

        # 输出匹配结果
        if matches:
            print(f"      [Graph Match] '{query_entity}' -> 映射为 -> '{matches[0]}'")
            return matches[0]
        return None

    # 支持多跳查询
    def get_context(self, entities: List[str], depth=2) -> str:
        found_nodes = []
        for e in entities:
            matched_node = self._fuzzy_match_node(e)
            if matched_node:
                found_nodes.append(matched_node)
        
        if not found_nodes:
            return ""

        result_lines = set()
        for node in found_nodes:
            # 获取以 node 为中心，半径为 depth 的子图
            # radius=1 找直接邻居，radius=2 找邻居的邻居
            try:
                subgraph = nx.ego_graph(self.graph, node, radius=depth)
                
                # 遍历子图中的边，生成文本
                for u, v, data in subgraph.edges(data=True):
                    rel = data.get('relation', 'related_to')
                    result_lines.add(f"- {u} {rel} {v}")
            except:
                pass
                
        return "\n".join(list(result_lines))

# 全局图实例
GLOBAL_GRAPH = EnhancedGraphDB()

# 定义状态节点，包含短期记忆消息列表、重写的搜索查询， 实体三元组关系，向量上下文，图谱上下文，融合后的最终上下文，以及待归档消息列表
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    search_query: str
    entities: List[str]
    vector_context: str
    graph_context: str
    final_context: str
    msgs_to_archive: List[BaseMessage]



# 根据State里面的message（短期记忆）重写问题，最终将重写的问题填到State的search_query字段里
def query_processing_node(state: AgentState, config: RunnableConfig = None):
    # 确保 config 不为 None
    config = config or {}
    
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
    rewritten = chain.invoke({"history": history_text, "question": last_user_msg})
    
    
    entities = []   
    conf = config.get("configurable", {})   # 获取 configurable 配置，默认为空字典
    
    # 实体提取，如果要查询图数据库就用这些实体来查询 (仅当开启图功能时执行，这里的prompt可以根据需要调整)
    if conf.get("use_graph_memory", True):
        entity_prompt = ChatPromptTemplate.from_messages([
            ("system", """提取关键实体（人名、作品名、地名、组织机构）。
            忽略抽象概念（如“难度”、“感觉”、“心情”）。
            返回 JSON 数组，例如 ['Alice', 'Python']。不要包含废话。"""),
            ("human", "{text}")
        ])

        # 调用api进行实体提取
        try:
            chain = entity_prompt | llm | JsonOutputParser()
            entities = chain.invoke(rewritten)
            if not isinstance(entities, list): entities = []
        except:
            print("[Warn] 实体提取失败，跳过。")
            entities = []

    print(f"\n[1.处理] 改写: {rewritten} | 实体: {entities}")
    return {"search_query": rewritten, "entities": entities}


# 通过上一个节点重写的问题去向量数据库里面进行检索，找到最相关的两条消息
def vector_retriever_node(state: AgentState, config: RunnableConfig = None):
    config = config or {}
    conf = config.get("configurable", {})
    
    # 消融实验开关
    if not conf.get("use_vector_memory", True):
        return {"vector_context": ""}
    
    # 获取上个节点改写的搜索查询，找到最相似的两条数据
    query = state["search_query"]
    results = VECTOR_STORE.similarity_search(query, k=2)
    
    # 构建上下文文本
    context = "\n".join([f"片段: {doc.page_content}" for doc in results])

    # 输出检索结果，便于调试
    if context:
        print(f"[2.向量] 命中 {len(results)} 条")
    else:
        print(f"[2.向量] 未命中任何内容。")
    return {"vector_context": context}


# 图数据库检索节点
def graph_retriever_node(state: AgentState, config: RunnableConfig = None):
    config = config or {}
    conf = config.get("configurable", {})
    
    # 消融实验开关
    if not conf.get("use_graph_memory", True):
        return {"graph_context": ""}
    
    # 获取实体列表，如果没有实体，直接返回空结果
    entities = state["entities"]
    if not entities:
        print(f"[2.图谱] 没有实体。")
        return {"graph_context": ""}
    
    # 查询图数据库，获取相关关系
    context = GLOBAL_GRAPH.get_context(entities, depth=1)

    # 输出检索结果，便于调试
    if context:
        print(f"[2.图谱] 命中关系:\n{context}")
    else:
        print(f"[2.图谱] 未命中任何关系。")

    return {"graph_context": context}


# 上下文融合节点，融合向量上下文和图谱上下文
def context_fusion_node(state: AgentState):
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

# 回复生成节点
def generator_node(state: AgentState):
    # 获取滑动窗口（短期记忆）和融合后的上下文（长期记忆）
    messages = state["messages"]
    context = state["final_context"]
    
    # 系统提示词
    system_prompt = """你是一个拥有【非结构化文本记忆】和【结构化知识图谱】的智能助手。
    请基于提供的上下文回答问题。
    
    上下文:
    {context}
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{messages}")
    ])

    # 打印调试信息
    print(f"\n[3.生成] 使用上下文:\n{context}")
    
    chain = prompt | llm
    response = chain.invoke({"context": context, "messages": messages})
    return {"messages": [response]}


# 内存管理路由：监控 Token是否超过了短期窗口大小
def memory_router_node(state: AgentState):
    messages = state["messages"]
    chat_msgs = [m for m in messages if not isinstance(m, SystemMessage)]
    
    if not chat_msgs:
        return {"msgs_to_archive": []}

    # 用encoding计算当前Token数
    current_tokens = sum([len(encoding.encode(m.content)) for m in chat_msgs])
    print(f"\n[4.监控] Token: {current_tokens}/{SHORT_TERM_TOKEN_LIMIT}")
    
    # 如果未超出限制，返回空列表
    if current_tokens <= SHORT_TERM_TOKEN_LIMIT:
        return {"msgs_to_archive": []}
    
    # 弹出旧消息（这里应该一弹弹两条出来）
    msgs_to_keep = list(chat_msgs)
    msgs_to_archive = []
    
    # 只要 Token 超标，且列表里至少有2条消息（保证能凑一对），就执行成对弹出
    while sum([len(encoding.encode(m.content)) for m in msgs_to_keep]) > SHORT_TERM_TOKEN_LIMIT:
        if len(msgs_to_keep) < 2:
            # 极端情况：只剩1条了但还是超标（比如用户发了一篇巨长的文章），应该不会有这种情况，因为现在里面还会有ai的回复
            if msgs_to_keep:
                msgs_to_archive.append(msgs_to_keep.pop(0))
            break

        # 正常情况：弹出最早的两个 (一般是 User, AI)
        m1 = msgs_to_keep.pop(0)
        m2 = msgs_to_keep.pop(0)
        
        msgs_to_archive.append(m1)
        msgs_to_archive.append(m2)
        
    return {"msgs_to_archive": msgs_to_archive}

# 写入向量数据库节点
def vector_archiver_node(state: AgentState, config: RunnableConfig = None):
    # 获取待归档消息列表
    msgs = state.get("msgs_to_archive", [])
    if not msgs: return {}
    
    # 检查是否使用向量数据库
    config = config or {}
    conf = config.get("configurable", {})
    if not conf.get("use_vector_memory", True):
        return {}
    
    print(f"   -> [Vector] 正在归档 {len(msgs)} 条消息...")
    
    # 1. 整理 User-AI 对 (Router 已经尽量保证成对传过来了，但这里再兜底处理一下)
    pairs = []
    buffer = []
    for m in msgs:
        buffer.append(m)
        if len(buffer) >= 2:
            if isinstance(buffer[0], HumanMessage) and isinstance(buffer[1], AIMessage):
                pairs.append(buffer[:2])
                buffer = buffer[2:]
            else:
                # 如果顺序乱了（比如连续两个 User），就放弃配对，单独存第一个
                pairs.append([buffer[0]])
                buffer = buffer[1:]

    # 处理剩余的
    for m in buffer: pairs.append([m])

    # 2. 查重并存储
    docs_to_add = []
    for pair in pairs:
        pair_text = "\n".join([f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}" for m in pair])

        # 生成摘要的 prompt(目前只存摘要，随时改成存原文)
        summary_prompt = f"""将以下对话转化为 1 句独立的陈述句事实。
        规则：
        1. 将“我”替换为“User”。
        2. 去除寒暄，只留干货。
        3. 字数控制在 100 字以内。
        
        对话：
        {pair_text}
        """
    
        new_summary = llm.invoke(summary_prompt).content.strip()

    
        # 查重: 检查库里是否已经有非常相似的
        existing = VECTOR_STORE.similarity_search_with_score(new_summary, k=1)
        if existing and existing[0][1] < 0.1: # 距离极小则跳过
            print(f"      [Vector Skip] 发现重复内容")
            continue
            
        chunks = text_splitter.split_text(pair_text)
        docs_to_add.extend(chunks)

    if docs_to_add:
        VECTOR_STORE.add_texts(docs_to_add)
        print(f"      [Vector Write] 存入 {len(docs_to_add)} 个切片")
    
    return {}

# 存入图数据库节点
def graph_archiver_node(state: AgentState, config: RunnableConfig = None):
    # 获取待归档消息列表
    msgs = state.get("msgs_to_archive", [])
    if not msgs: return {}
    
    # 检查是否使用图数据库
    config = config or {}
    conf = config.get("configurable", {})
    if not conf.get("use_graph_memory", True):
        return {}
    
    text_content = "\n".join([m.content for m in msgs])
    
    # 生成三元组提取的 prompt
    extract_prompt = """你是一个知识图谱构建专家。请从以下文本中提取明确的事实，格式化为 (主体, 关系, 客体) 的三元组 JSON 列表。
    规则：
    1. 实体应尽量标准化（如“我”转为“用户”），且不带有标点符号：“。，《》……” 等
    2. 关系应简洁。
    3. 只提取有价值的长期事实。
    
    文本:
    {text}
    
    输出示例:
    [["用户", "职业", "程序员"], ["Python", "属于", "编程语言"]]
    """
    
    # 执行三元组提取
    try:
        print(f"   -> [Graph] 正在提取知识三元组...")
        chain = ChatPromptTemplate.from_template(extract_prompt) | llm | JsonOutputParser()
        triples = chain.invoke({"text": text_content})
        
        if isinstance(triples, list) and len(triples) > 0:
            GLOBAL_GRAPH.add_triples(triples)
        else:
            print("      [Graph] 未提取到有效三元组。")
            
    except Exception as e:
        print(f"      [Graph] 提取出错: {e}")
        
    return {}


# 清理归档消息节点
def cleanup_node(state: AgentState):
    msgs = state.get("msgs_to_archive", [])
    if not msgs: return {}
    
    return {"messages": [RemoveMessage(id=m.id) for m in msgs]}



# 构建agent的流程图
workflow = StateGraph(AgentState)

# 1. 添加节点
workflow.add_node("query_processing", query_processing_node)
workflow.add_node("vector_retriever", vector_retriever_node)
workflow.add_node("graph_retriever", graph_retriever_node)
workflow.add_node("context_fusion", context_fusion_node)
workflow.add_node("generator", generator_node)
workflow.add_node("memory_router", memory_router_node)
workflow.add_node("vector_archiver", vector_archiver_node)
workflow.add_node("graph_archiver", graph_archiver_node)
workflow.add_node("cleanup", cleanup_node)

# 2. 定义边
workflow.set_entry_point("query_processing")
workflow.add_edge("query_processing", "vector_retriever")
workflow.add_edge("vector_retriever", "graph_retriever")
workflow.add_edge("graph_retriever", "context_fusion")
workflow.add_edge("context_fusion", "generator")
workflow.add_edge("generator", "memory_router")

# 条件边
def should_archive(state: AgentState):
    if state["msgs_to_archive"]:
        return "archive"
    return "end"

workflow.add_conditional_edges(
    "memory_router",
    should_archive,
    {
        "archive": "vector_archiver",
        "end": END
    }
)

workflow.add_edge("vector_archiver", "graph_archiver")
workflow.add_edge("graph_archiver", "cleanup")
workflow.add_edge("cleanup", END)

# 3. 编译
conn = sqlite3.connect("checkpoints_v7_fixed.db", check_same_thread=False)
memory = SqliteSaver(conn)
app = workflow.compile(checkpointer=memory)


# ================= 测试运行 =================
def run_chat(user_input, thread_id, enable_vector=True, enable_graph=True):
    print(f"\n{'='*20} User: {user_input} {'='*20}")
    
    config = {
        "configurable": {
            "thread_id": thread_id,
            "use_vector_memory": enable_vector,
            "use_graph_memory": enable_graph
        }
    }
    
    # 第一次对话时，graph.stream 会处理消息初始化
    events = app.stream(
        {"messages": [HumanMessage(content=user_input)]}, 
        config, 
        stream_mode="values"
    )
    
    final_msg = None
    for event in events:
        if "messages" in event:
            msgs = event["messages"]
            if msgs:
                final_msg = msgs[-1]
            
    if isinstance(final_msg, AIMessage):
        print(f"\n>>> AI: {final_msg.content}")


if __name__ == "__main__":
    tid = str(uuid.uuid4())
    print(f"Thread ID: {tid}")
    
    print("\n--- 阶段1: 建立记忆 (全开启) ---")
    # 这句话包含实体关系：黑暗之魂 -> 制作人 -> 宫崎英高
    run_chat("我最近特别沉迷《黑暗之魂》，它是宫崎英高做的游戏，难度很高。", tid, True, True)
    
    # 灌水触发归档
    run_chat("凑字数 " * 40, tid, True, True) 
    
    print("\n--- 阶段2: 验证检索能力 ---")
    
    
    run_chat("我刚才提到的那个游戏的制作人是谁？", tid, enable_vector=True, enable_graph=True)
    
    run_chat("你刚才提到的黑暗之魂有哪些特点？", tid, enable_vector=True, enable_graph=True)

    print("\n✅ 测试完成。")