import json
import random
import os
import uuid
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser

# ================= 配置 =================
OUTPUT_FILE = "experiment_dataset_hard.json"
NUM_SAMPLES = 20        # 总样本数
DISTRACTOR_MSG_COUNT = 8 # 干扰消息数量 (建议 8-12)
DISTRACTOR_LEN = 150     # 干扰消息长度

# LLM 初始化
llm = ChatOpenAI(
    model="qwen-plus", # 建议用强力模型生成数据，如 qwen-max 或 gpt-4
    temperature=0.8,
    openai_api_key="sk-2770a3f619c14f31a87d47924de34af2",
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 读取背景噪音
if not os.path.exists("background.txt"):
    BACKGROUND_TEXT = "人工智能发展迅速，深度学习是其中的核心技术..." * 100
else:
    with open("background.txt", "r", encoding="utf-8") as f:
        BACKGROUND_TEXT = f.read()

def get_noise(length=100):
    """获取纯背景噪音"""
    if len(BACKGROUND_TEXT) < length: return BACKGROUND_TEXT
    start = random.randint(0, len(BACKGROUND_TEXT) - length - 10)
    return BACKGROUND_TEXT[start:start+length].replace("\n", " ")

# ================= 数据结构定义 =================

class MultiHopCase(BaseModel):
    """多跳推理类型：需要结合两个事实"""
    fact_1: str = Field(description="第一个事实，例如：'A是B的父亲'")
    fact_2: str = Field(description="第二个事实，例如：'B是C的老师'")
    question: str = Field(description="需要结合两者的提问，例如：'A的孩子从事什么职业？'")
    answer: str = Field(description="标准答案")

class AdversarialCase(BaseModel):
    """对抗干扰类型：包含事实和混淆项"""
    true_fact: str = Field(description="真实的事实，例如：'密码是1234'")
    fake_fact: str = Field(description="干扰性极强的事实，例如：'旧密码是1234但已过期' 或 '管理员的ID是1234'")
    question: str = Field(description="提问")
    answer: str = Field(description="标准答案")

# ================= 生成逻辑 =================

def generate_dataset_hard():
    print(f"🚀 开始生成高难度对抗数据集 (共 {NUM_SAMPLES} 条)...")
    dataset = []

    # 1. 定义 Prompt 模板
    
    # 多跳 Prompt
    prompt_multi = ChatPromptTemplate.from_template("""
    请生成一个【多跳推理】测试用例。
    要求：
    1. 事实必须是虚构的（科幻/魔幻/谍战背景）。
    2. 答案必须依赖两个事实才能推导出来，缺一不可。
    3. 两个事实不要在语义上过于接近，最好涉及不同的人物或地点。
    
    {format_instructions}
    """)
    
    # 对抗 Prompt
    prompt_adv = ChatPromptTemplate.from_template("""
    你是一个专攻【大模型对抗攻击】的数据集生成专家。你需要生成一组非常难以区分的“事实 vs 干扰”数据。

    请严格按照以下步骤生成：

    Step 1: 确定一个【核心实体】（如某个人名、地点、计划代号）。
    Step 2: 设计一个【真实事实 (true_fact)】，描述该实体的当前状态。
    Step 3: 设计一个【干扰事实 (fake_fact)】。要求：
        - 必须包含【核心实体】的名称（确保向量相似度极高）。
        - 必须与真实事实在语义上冲突。
        - 采用以下三种攻击模式之一：
            A. 【时序过期模式】: 干扰事实是“旧的/过期的”信息。
            (例: 真="密码现在是999"; 假="密码上周还是000")
            B. 【否定/取消模式】: 干扰事实是“被否决/取消”的计划。
            (例: 真="我们最终选择了B方案"; 假="A方案原本是首选但被废弃了")
            C. 【主体混淆模式】: 描述极其相似的另一个人的状态。
            (例: 真="特工007的代号是鹰"; 假="特工006的代号是鹰")

    Step 4: 基于【真实事实】生成问题。

    ---
    【Few-Shot 示例】:
    1. 
    true_fact: "蓝宝石号飞船的发射代码是 Alpha-9。"
    fake_fact: "蓝宝石号飞船原本的预设代码是 Beta-1，但后来废弃了。"
    question: "蓝宝石号飞船的最终发射代码是什么？"
    answer: "Alpha-9"

    2.
    true_fact: "现任财务主管是 Sarah Connor。"
    fake_fact: "John Connor 曾担任财务主管，但他上个月离职了。"
    question: "现在的财务主管是谁？"
    answer: "Sarah Connor"
    ---

    请输出 JSON 格式:
    {format_instructions}
    """)

    parser_multi = PydanticOutputParser(pydantic_object=MultiHopCase)
    parser_adv = PydanticOutputParser(pydantic_object=AdversarialCase)

    for i in range(NUM_SAMPLES):
        # 随机选择一种模式：50% 多跳，50% 对抗
        mode = "multihop" if random.random() < 0.5 else "adversarial"
        
        try:
            item_data = {}
            distractor_msgs = []
            
            # 先填充一些背景噪音作为底料
            for _ in range(DISTRACTOR_MSG_COUNT):
                distractor_msgs.append(get_noise(DISTRACTOR_LEN))

            if mode == "multihop":
                # === 生成多跳数据 ===
                chain = prompt_multi | llm | parser_multi
                res = chain.invoke({"format_instructions": parser_multi.get_format_instructions()})
                
                # 策略：埋藏位置不变，但去掉【标签】
                # 可以加一点点自然的口语前缀，让它混在小说里不那么突兀，也可以直接放
                
                idx1, idx2 = 0, len(distractor_msgs) // 2
                
                # 修改前：distractor_msgs.insert(idx1, f"【线索A】{res.fact_1}")
                # 修改后：直接放入，或者加自然前缀
                distractor_msgs.insert(idx1, f"顺便提一下，{res.fact_1}") 
                distractor_msgs.insert(idx2, f"还有件事忘了说，{res.fact_2}")
                
                item_data = {
                    "category": "多跳推理 (Multi-hop)",
                    "fact_content": f"{res.fact_1} | {res.fact_2}", 
                    "question": res.question,
                    "ground_truth": res.answer
                }

            else:
                # === 生成对抗数据 ===
                chain = prompt_adv | llm | parser_adv
                res = chain.invoke({"format_instructions": parser_adv.get_format_instructions()})
                
                # 策略：真实事实放在开头，对抗事实放在结尾
                
                # 修改前：distractor_msgs.insert(0, f"【重要记录】{res.true_fact}")
                # 修改前：distractor_msgs.insert(-1, f"【闲聊干扰】{res.fake_fact}")

                # 修改后：
                distractor_msgs.insert(0, f"你需要记住，{res.true_fact}")
                
                # 对抗样本如果不加标签，就更具迷惑性！
                # 比如：true="密码是1234"，fake="以前密码是9999"
                # 如果没有标签，Agent 必须依靠语义的时间/状态判断，这才是真正的高难度
                distractor_msgs.insert(-1, f"哎不对，我想起来{res.fake_fact}") 
                
                item_data = {
                    "category": "语义对抗 (Adversarial)",
                    "fact_content": res.true_fact,
                    "question": res.question,
                    "ground_truth": res.answer
                }

            # 组装最终 JSON 对象
            # 注意：这里的结构微调了，我们将所有的 "history" 都放在 distractor_messages 里
            # 你的 agent.chat 需要遍历这个列表发送消息
            final_item = {
                "id": str(uuid.uuid4()),
                "category": item_data["category"],
                "fact": "", # 留空，因为事实已经混入 messages 了
                "distractor_messages": distractor_msgs, 
                "question": item_data["question"],
                "ground_truth": item_data["ground_truth"]
            }
            dataset.append(final_item)

        except Exception as e:
            print(f"  ⚠️ 生成失败: {e}")
            continue

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 高难度数据集已生成: {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_dataset_hard()