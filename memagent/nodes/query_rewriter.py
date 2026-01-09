#!/usr/bin/env python3
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
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
        """
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "历史:\n{history}\n\n当前输入: {question}\n\n改写结果:")
        ])
        self.chain = self.prompt_template | self.llm | StrOutputParser()
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """执行查询重写"""
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
        
        # 返回改写后的查询，并清空retrieved_context
        return {"search_query": rewritten_query, "retrieved_context": ""}