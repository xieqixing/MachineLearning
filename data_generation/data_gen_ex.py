import json
import random
import os
import uuid
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# ================= 配置 =================
OUTPUT_FILE = "experiment_dataset_trap.json"
NUM_SAMPLES = 20        # 样本数
NOISE_COUNT = 8         # 干扰条数

# LLM 初始化
llm = ChatOpenAI(
    model="qwen-plus", 
    temperature=0.8,
    openai_api_key="your_api_key_here",
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 读取背景文本 (用于填充空隙)
if not os.path.exists("background.txt"):
    BACKGROUND_TEXT = "在这个赛博朋克的世界里，数据流如同血液般流淌..." * 100
else:
    with open("background.txt", "r", encoding="utf-8") as f:
        BACKGROUND_TEXT = f.read()

def get_filler(length=80):
    start = random.randint(0, len(BACKGROUND_TEXT) - length - 10)
    return BACKGROUND_TEXT[start:start+length].replace("\n", " ")

# ================= 数据结构 =================

class TrapCase(BaseModel):
    """
    代号陷阱用例：
    1. Alias Link: A 是 B
    2. True Fact: B 发生了 C (不提 A)
    3. Distractors: A 发生了 D, E, F (全是干扰)
    """
    codename: str = Field(description="代号/别名，例如：'幽灵'、'X计划'")
    real_name: str = Field(description="真实实体名，例如：'约翰·道'、'阿波罗引擎'")
    attribute: str = Field(description="核心属性，例如：'藏在地下室'、'启动密码是123'")
    
    # 干扰项必须包含 codename，看起来非常像答案
    distractor_info: str = Field(description="关于代号的错误信息，例如：'幽灵据说在屋顶'，不要包含真实实体名")
    
    question: str = Field(description="提问，必须使用代号(Codename)进行提问")
    answer: str = Field(description="标准答案")

# ================= 生成逻辑 =================

def generate_dataset_trap():
    print(f"🚀 正在生成【Vector杀手】数据集 (共 {NUM_SAMPLES} 条)...")
    
    prompt = ChatPromptTemplate.from_template("""
    你是一个对抗性数据生成专家。请设计一个【代号分离陷阱】测试用例。
    
    原理：
    我们将把“问题的主语”和“答案的主语”故意分开，中间用一个“身份揭秘”连接。
    
    生成步骤：
    1. 设计一个【代号】(Codename) 和一个【真名】(Real Name)。
    2. 设计一个事实：【真名】具有某个属性 (Attribute)。注意：这句话里绝对不能出现【代号】！
    3. 设计干扰信息：描述【代号】的一些错误属性。这句话要包含【代号】，让向量检索误以为它是答案。
    4. 提问：问【代号】的属性是什么？

    示例：
    Codename: "毒蛇"
    Real Name: "安娜医生"
    Attribute: "正在前往巴黎" (注意：只提安娜，不提毒蛇)
    Distractor: "有人看见毒蛇在伦敦，也有情报说毒蛇在柏林。"
    Question: "毒蛇正在前往哪里？"
    Answer: "巴黎"

    请输出 JSON:
    {format_instructions}
    """)
    
    parser = PydanticOutputParser(pydantic_object=TrapCase)
    
    dataset = []

    for i in range(NUM_SAMPLES):
        try:
            chain = prompt | llm | parser
            res = chain.invoke({"format_instructions": parser.get_format_instructions()})
            
            distractor_msgs = []
            
            # -------------------------------------------------------------
            # 💀 核心陷阱布局 (The Trap Layout)
            # -------------------------------------------------------------
            
            # 1. 干扰层 (High Similarity, Wrong Answer)
            # 这些消息包含 "Question Keyword" (Codename)，会被 Vector 优先检索
            # 我们生成多条类似的干扰
            distractor_msgs.append(f"关于{res.codename}的行踪，目前的假消息很多。{res.distractor_info}")
            distractor_msgs.append(f"情报局拦截到消息：{res.codename}似乎并未离开大本营。")
            distractor_msgs.append(f"注意：所有关于{res.codename}的目击报告都需要二次核实。")
            
            # 插入一些纯背景填充
            for _ in range(3):
                distractor_msgs.append(get_filler())

            # 2. 身份层 (Link)
            # 必须有一句话建立连接。
            # 放在中间位置，模拟偶尔提到的线索
            link_msg = f"绝密档案解锁：经过比对，目标【{res.codename}】的真实身份确认为【{res.real_name}】。"
            distractor_msgs.insert(2, link_msg)
            
            # 3. 真相层 (Low Similarity, True Answer)
            # 这句话只包含 Real Name，不含 Codename。
            # Vector 检索 "Question(Codename)" 时，这句话的得分为 0 或极低。
            # 只有 Graph 知道 Codename == Real Name 才能关联到这里。
            true_msg = f"最新监控显示，{res.real_name} {res.attribute}。"
            
            # 将真相藏在比较深的地方（或者随机位置），但为了实验效果，建议不要放最后
            # 放中间偏后，确保不在 Context Window 的最前沿（防止运气好碰上）
            distractor_msgs.insert(len(distractor_msgs)-2, true_msg)

            item = {
                "id": str(uuid.uuid4()),
                "category": "实体跳跃陷阱 (Entity-Hop Trap)",
                "fact_content": f"{res.codename} == {res.real_name} -> {res.attribute}",
                "distractor_messages": distractor_msgs,
                "question": res.question,
                "ground_truth": res.answer
            }
            
            dataset.append(item)
            print(f"  [{i+1}] Q: {res.question}")
            print(f"      Trap: 真相是关于 '{res.real_name}' 的，但问题问的是 '{res.codename}'")

        except Exception as e:
            print(f"  ⚠️ Error: {e}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 数据集生成完毕: {OUTPUT_FILE}")
    print("💡 预期结果: Vector 模式检索不到含有答案的 'true_msg'，因为它只含真名不含代号。")

if __name__ == "__main__":
    generate_dataset_trap()