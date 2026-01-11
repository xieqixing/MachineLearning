#!/usr/bin/env python3
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


class GeneratorNode:
    """生成回复节点"""
    
    def __init__(self, llm: ChatOpenAI, verbose: bool = True):
        self.llm = llm
        self.verbose = verbose
        self._setup_prompts()
    
    def _setup_prompts(self):
        """设置提示模板"""
        self.system_tmpl = """你是一个拥有【非结构化文本记忆】和【结构化知识图谱】的智能助手。
    请基于提供的上下文回答问题。
        
        上下文:
        {context}
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_tmpl ),
            ("placeholder", "{messages}")
        ])
        self.chain = prompt | self.llm
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # 获取滑动窗口（短期记忆）和融合后的上下文（长期记忆）
        messages = state["messages"]
        context = state["final_context"]
        
        # 生成回复
        response = self.chain.invoke({
            "context": context, 
            "messages": messages
        })
        
        # 生成回复，append到消息列表里面去
        return {"messages": [response]}