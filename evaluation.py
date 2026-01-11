import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from pathlib import Path
import time
import json
import uuid
import wandb
import numpy as np
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# 导入LLM agent
from memagent import MemoryAgent, MemoryAgentConfig

# ================= 配置 =================
DATASET_FILE = "experiment_dataset.json"
PROJECT_NAME = "LLM-Memory-System-Final"

# 用作裁判评分的LLM
eval_llm = ChatOpenAI(
    model="qwen-plus", 
    temperature=0,
    openai_api_key="your_api_key_here",
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 裁判打分：返回 True/False 和 分数
def llm_judge(question, answer, truth):
    prompt = f"""
    标准答案: {truth}
    AI 回答: {answer}
    问题: {question}
    
    请判断 AI 回答是否包含了标准答案的核心意思。
    输出 JSON: {{"correct": true/false, "score": 1-5}}
    """
    try:
        res = eval_llm.invoke(prompt).content
        if "```" in res: res = res.split("```json")[-1].split("```")[0]
        data = json.loads(res)
        return data["correct"], data["score"]
    except:
        return False, 0


# 运行评测函数
def run_evaluation():
    # 加载数据
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    # 初始化 WandB
    wandb.init(project=PROJECT_NAME, name="Comprehensive-Eval-v1")
    
    # 定义 WandB 表格列
    columns = [
        "Config", "Category", "Question", "Answer", "Truth", 
        "Correct", "Score", "Latency(s)", 
        "Vector Hit", "Graph Hit"
    ]
    table = wandb.Table(columns=columns)
    
    # 定义对比实验组消融实验）
    configs = [
        {"name": "Baseline (No Mem)", "vec": False, "graph": False},
        {"name": "Vector Only",       "vec": True,  "graph": False},
        {"name": "Graph Only",        "vec": False, "graph": True},
        {"name": "Hybrid (Full)",     "vec": True,  "graph": True},
    ]


    print(f"🚀 开始评测，共 {len(dataset)} 个样本 x {len(configs)} 种配置")

    # 所有评测产物放这里
    BASE_RUN_DIR = Path("./eval_runs")  

    # 循环遍历所有的测试
    for conf in configs:
        print(f"\n--- Running Configuration: {conf['name']} ---")
        
        metrics = {
            "latency": [], "score": [], "accuracy": [],
            "vector_hit_rate": [], "graph_hit_rate": []
        }

        for i, item in enumerate(dataset):
            thread_id = f"{conf['name'].replace(' ', '_')}_{i}_{uuid.uuid4().hex[:8]}"
            run_dir = BASE_RUN_DIR / thread_id
            run_dir.mkdir(parents=True, exist_ok=True)

            # 创建Agent配置
            config = MemoryAgentConfig(
                verbose=True,  # 显示详细日志
                vector_store_path=str(run_dir / "chroma"),              # 每次一个全新向量库目录
                checkpoints_db=str(run_dir / "checkpoints.sqlite")      # 每次一个全新checkpoint库
            )
            
            # 初始化Agent
            agent = MemoryAgent(config)
            
            # 记忆植入
            if item.get("fact"):
                agent.chat(
                    item["fact"], 
                    thread_id=thread_id,
                    enable_vector=conf["vec"], 
                    enable_graph=conf["graph"]
                )
            
            # 多轮干扰 (关键步骤)
            # 这一步会多次调用 Agent，模拟时间流逝和上下文滑动
            for dist_msg in item["distractor_messages"]:
                agent.chat(
                    dist_msg, 
                    thread_id=thread_id, 
                    enable_vector=conf["vec"], 
                    enable_graph=conf["graph"]
                )
                
            # 提问与测试
            start_time = time.time()
            
            # 获取 Final State 以检查 Context
            final_state = agent.chat(
                item["question"], 
                thread_id=thread_id, 
                enable_vector=conf["vec"], 
                enable_graph=conf["graph"]
            )
            
            end_time = time.time()
            latency = end_time - start_time

            # 关闭agent
            agent.close()
            
            # 解析结果
            ai_msg = final_state["messages"][-1].content
            print(ai_msg)
            
            # 检查 State 中的 context 是否为空
            vector_hit = 1 if len(final_state.get("vector_context", "")) > 10 else 0
            graph_hit = 1 if len(final_state.get("graph_context", "")) > 10 else 0
            
            # LLM 裁判
            is_correct, score = llm_judge(item["question"], ai_msg, item["ground_truth"])
            
            # 记录数据
            metrics["latency"].append(latency)
            metrics["score"].append(score)
            metrics["accuracy"].append(1 if is_correct else 0)
            metrics["vector_hit_rate"].append(vector_hit)
            metrics["graph_hit_rate"].append(graph_hit)
            
            # 添加到 WandB 表格
            table.add_data(
                conf["name"], item["category"], item["question"], ai_msg, item["ground_truth"],
                is_correct, score, round(latency, 2), vector_hit, graph_hit
            )
            
            print(f"   [{i+1}/{len(dataset)}] Q: {item['question'][:15]}... | Correct: {is_correct} | V-Hit: {vector_hit} | G-Hit: {graph_hit}")

        # 计算该配置的平均指标并 Log
        wandb.log({
            f"{conf['name']}/avg_latency": np.mean(metrics["latency"]),
            f"{conf['name']}/avg_score": np.mean(metrics["score"]),
            f"{conf['name']}/accuracy": np.mean(metrics["accuracy"]),
            f"{conf['name']}/vector_hit_rate": np.mean(metrics["vector_hit_rate"]),
            f"{conf['name']}/graph_hit_rate": np.mean(metrics["graph_hit_rate"]),
        })

    wandb.log({"Evaluation Details": table})
    wandb.finish()
    print("\n所有评测完成！请前往 WandB 查看可视化报告。")

if __name__ == "__main__":
    run_evaluation()