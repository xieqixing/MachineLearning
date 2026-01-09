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
        self.system_tmpl = """你是一个助手。请基于【长期记忆】和【当前对话】回答。
        
        【长期记忆】:
        {context}
        
        注意：
        1. 长期记忆中的信息优先于你的通用知识。
        2. 如果用户询问具体原话，请参考括号中的"来源参考"。
        """
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_tmpl),
            ("placeholder", "{messages}"),
        ])
        self.chain = self.prompt_template | self.llm
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """生成回复"""
        # 获取短期记忆和搜索到的长期记忆
        messages = state["messages"]
        context = state["retrieved_context"]
        
        # 生成回复
        response = self.chain.invoke({
            "context": context, 
            "messages": messages
        })
        
        return {"messages": [response]}