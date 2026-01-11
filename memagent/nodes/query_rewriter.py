#!/usr/bin/env python3
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI


class QueryRewriterNode:
    """查询重写节点"""
    
    def __init__(self, llm: ChatOpenAI, verbose: bool = True):
        self.llm = llm
        self.verbose = verbose
        self._setup_prompts()
    
    def _setup_prompts(self):
        """设置提示模板"""
        self.system_prompt = """你是一个查询改写工具。
        任务：结合历史，将用户的最新输入改写为独立、完整的搜索语句。
        规则：
        1. 补全主语和指代词（如"它"->具体名词）。
        2. 严禁回答问题。
        3. 如果无需改写，原样返回。
        4. 如果含有大量的无意义信息，可以适当压缩信息。
        """
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "历史:\n{history}\n\n当前输入: {question}\n\n改写结果:")
        ])
        self.chain = self.prompt_template | self.llm | StrOutputParser()
    
    # 执行查询重写
    def __call__(self, state: Dict[str, Any], config: RunnableConfig = None) -> Dict[str, Any]:
        # 确保 config 不为 None
        config = config or {}

        # 获取状态里的消息列表
        messages = state["messages"]
        last_user_msg = messages[-1].content
        history = messages[:-1]
        
        # 构建历史文本
        history_text = "\n".join([f"{m.type}: {m.content}" for m in history])
        
        # 执行查询改写
        rewritten_query = self.chain.invoke({
            "history": history_text, 
            "question": last_user_msg
        })
        
        if self.verbose:
            print(f"\n[1.改写] '{last_user_msg}' -> '{rewritten_query}'")

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
                chain = entity_prompt | self.llm | JsonOutputParser()
                entities = chain.invoke(rewritten_query)
                if not isinstance(entities, list): entities = []
            except:
                print("[Warn] 实体提取失败，跳过。")
                entities = []

        if self.verbose:
            print(f"\n[1.处理] 改写: {rewritten_query} | 实体: {entities}")
        
        # 返回改写后的查询, 以及提取的实体
        return {"search_query": rewritten_query, "entities": entities}