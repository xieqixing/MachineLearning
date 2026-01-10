from langchain_openai import ChatOpenAI
from ..graphDB import EnhancedGraphDB
from typing import Dict, Any
from langchain_core.runnables import RunnableConfig
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate


# 存入图数据库节点
class GraphArchiverNode:
    def __init__(self, llm: ChatOpenAI, graphDB: EnhancedGraphDB, verbose: bool = True):
        self.llm = llm
        self.graphDB = graphDB
        self.verbose = verbose

    def __call__(self, state: Dict[str, Any], config: RunnableConfig = None) -> Dict[str, Any]:
       # 获取待归档消息列表
        msgs = state.get("msgs_to_archive", [])
        if not msgs: return {}
        
        # 检查是否使用图数据库
        config = config or {}
        conf = config.get("configurable", {})
        if not conf.get("use_graph_memory", True):
            return {}
        
        # print("DEBUG msgs type:", type(msgs))
        # print("DEBUG len(msgs):", len(msgs))
        # print("DEBUG first item type:", type(msgs[0]))
        # print("DEBUG first item:", msgs[0])

        text_content = "\n".join([m.content for m in msgs])
        
        # 生成三元组提取的 prompt
        extract_prompt = """你是一个知识图谱构建专家。请从以下文本中提取明确的事实，格式化为 (主体, 关系, 客体) 的三元组 JSON 列表。
        规则：
        1. 实体应尽量标准化（如“我”转为“用户”），且不带有标点符号：“。，《》……” 等
        2. 关系应简洁，除了极端情况，一定要控制在4个字以内。
        3. 只提取有价值的长期事实。
        
        文本:
        {text}
        
        输出示例:
        [["用户", "职业", "程序员"], ["Python", "属于", "编程语言"]]
        """
        
        # 执行三元组提取
        try:
            print(f"   -> [Graph] 正在提取知识三元组...")
            chain = ChatPromptTemplate.from_template(extract_prompt) | self.llm | JsonOutputParser()
            triples = chain.invoke({"text": text_content})
            
            if isinstance(triples, list) and len(triples) > 0:
                self.graphDB.add_triples(triples)
            else:
                print("      [Graph] 未提取到有效三元组。")
                
        except Exception as e:
            print(f"      [Graph] 提取出错: {e}")
            
        return {}