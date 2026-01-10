import json
import random
import os
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

# ================= 配置 =================
OUTPUT_FILE = "experiment_dataset.json"
NUM_SAMPLES = 2  # 生成多少组测试数据
DISTRACTOR_MSG_COUNT = 10  # 干扰消息的条数 (确保足够把事实挤出窗口)
DISTRACTOR_MSG_LEN = 100   # 每条干扰消息大概多少字

# LLM 初始化
llm = ChatOpenAI(
    model="qwen-plus", 
    temperature=0.7,
    openai_api_key="sk-2770a3f619c14f31a87d47924de34af2", 
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 读取背景文本
if not os.path.exists("background.txt"):
    # 如果没有文件，生成假数据
    BACKGROUND_TEXT = "人工智能（AI）是计算机科学的一个分支，它致力于创造能够执行通常需要人类智能的任务的机器..." * 500
else:
    with open("background.txt", "r", encoding="utf-8") as f:
        BACKGROUND_TEXT = f.read()

def get_random_distractor_chunk(length):
    """从背景文本中随机截取一段"""
    if len(BACKGROUND_TEXT) < length:
        return BACKGROUND_TEXT
    start = random.randint(0, len(BACKGROUND_TEXT) - length - 1)
    return BACKGROUND_TEXT[start : start + length].replace("\n", " ")

def generate_dataset():
    print(f"🚀 正在生成 {NUM_SAMPLES} 组多轮对话测试数据...")
    
    # 定义三种记忆类型
    categories = ["实体细节 (Entity)", "关系推理 (Relation)", "时序数字 (Numeric)"]
    
    # Prompt: 让 LLM 生成事实和问题
    prompt = ChatPromptTemplate.from_template("""
    你是一个数据集生成专家。请生成一组用于测试 AI 长期记忆的问答数据。
    
    测试类型: {category}
    
    要求：
    1. "fact": 一个独立的陈述句，包含具体的虚构事实（不要用真实世界常识）。
    2. "question": 针对该事实的提问。
    3. "answer": 简短的标准答案。
    
    输出 JSON 格式:
    {{
        "fact": "...",
        "question": "...",
        "answer": "..."
    }}
    """)
    
    chain = prompt | llm | JsonOutputParser()
    dataset = []

    for i in range(NUM_SAMPLES):
        cat = random.choice(categories)
        try:
            # 1. 生成核心事实
            res = chain.invoke({"category": cat})
            
            # 2. 生成多条干扰消息 (模拟多轮闲聊)
            distractors = []
            for _ in range(DISTRACTOR_MSG_COUNT):
                # 随机截取一段文本，并加上一点前缀让它看起来像对话
                chunk = get_random_distractor_chunk(DISTRACTOR_MSG_LEN)
                distractors.append(chunk)
                
            item = {
                "id": f"test_{i:03d}",
                "category": cat,
                "fact": res["fact"],
                "distractor_messages": distractors, # 这是一个列表
                "question": res["question"],
                "ground_truth": res["answer"]
            }
            dataset.append(item)
            print(f"  [{i+1}/{NUM_SAMPLES}] {cat}: {res['question']}")
            
        except Exception as e:
            print(f"  [Error] {e}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 数据集已保存至 {OUTPUT_FILE}")
    print(f"   结构: 1条事实 -> {DISTRACTOR_MSG_COUNT}条干扰对话 -> 1个提问")

if __name__ == "__main__":
    generate_dataset()