from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from typing import Dict, Any
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage


# 写入向量数据库节点
class VectorArchiverNode:
    def __init__(self, llm: ChatOpenAI, vector_store: Chroma, verbose: bool = True):
        self.llm = llm
        self.vector_store = vector_store
        self.verbose = verbose

    def __call__(self, state: Dict[str, Any], config: RunnableConfig = None) -> Dict[str, Any]:
        
        # 获取待归档消息列表
        msgs = state.get("msgs_to_archive", [])
        if not msgs: return {}
        
        # 检查是否使用向量数据库
        config = config or {}
        conf = config.get("configurable", {})
        if not conf.get("use_vector_memory", True):
            return {}
        

        # raw_text = "\n".join([f"{m.type}: {m.content}" for m in msgs])
        raw_text = "\n".join([f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}" for m in msgs])
        
        # 生成摘要
        summary_prompt = f"""将以下对话转化为 1 句独立的陈述句事实。
        规则：
        1. 将"我"替换为"User"。
        2. 去除寒暄，只留干货。
        3. 字数控制在 60 字以内。
        
        对话：
        {raw_text}
        """
        
        new_summary = self.llm.invoke(summary_prompt).content.strip()
        
        if self.verbose:
            print(f"\n[4.检测] 准备入库: {new_summary}")
        
        # 重复检测
        existing_docs = self.vector_store.similarity_search_with_score(
            new_summary, 
            k=1
        )
        
        is_duplicate = False
        if existing_docs:
            doc, score = existing_docs[0]
            if score < 0.35:  # Chroma的score是L2距离
                if self.verbose:
                    print(f"   -> 发现重复 (Distance={score:.4f}): '{doc.page_content}'")
                    print("   -> 跳过写入。")
                is_duplicate = True
                
        # 不重复则写入向量库
        if not is_duplicate:
            if self.verbose:
                print("   -> 写入向量库。")
           
            self.vector_store.add_texts(
                texts=[new_summary],
                metadatas=[{
                    "raw_content": raw_text, 
                }] 
            )
        
        # 直接返回
        return {}
    