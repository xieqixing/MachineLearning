#!/usr/bin/env python3
import os
import uuid

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from memagent import MemoryAgent, MemoryAgentConfig


def main():
    # 创建Agent配置
    config = MemoryAgentConfig(
        verbose=True,  # 显示详细日志
        checkpoints_db="memory_agent_checkpoints.db"
    )
    
    # 初始化Agent
    agent = MemoryAgent(config)
    
    try:
        tid = str(uuid.uuid4())
        # 示例1: 创建新对话
        print("=== 示例1: 新对话 ===")
        response = agent.chat("我最近特别沉迷《黑暗之魂》，它是宫崎英高做的游戏，难度很高。", thread_id=tid)
        print(f"AI回复: {response}")
        
        
        response = agent.chat("凑字数 " * 40, thread_id=tid)
        print(f"AI回复: {response}")

        response = agent.chat("我叫王五，你叫什么？", thread_id=tid)
        print(f"AI回复: {response}")

        
        response = agent.chat("我刚才提到的那个游戏的制作人是谁？", thread_id=tid)
        print(f"AI回复: {response}")

        response = agent.chat("你刚才提到的黑暗之魂有哪些特点？", thread_id=tid)
        print(f"AI回复: {response}")
        
        # 获取记忆统计
        print("\n=== 记忆系统统计 ===")
        stats = agent.get_memory_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")
            
    finally:
        # 关闭Agent
        agent.close()

        # 删除所有数据库
        agent.reset_all_storages()


if __name__ == "__main__":
    main()