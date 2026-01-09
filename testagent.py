#!/usr/bin/env python3
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
        # 示例1: 创建新对话
        print("=== 示例1: 新对话 ===")
        response = agent.chat("你好，我叫王五。")
        print(f"AI回复: {response}")
        
        # 示例2: 继续对话
        print("\n=== 示例2: 继续对话 ===")
        response = agent.chat("我是 Python 程序员。")
        print(f"AI回复: {response}")
        
        # 示例3: 测试记忆功能
        print("\n=== 示例3: 测试记忆功能 ===")
        response = agent.chat("我刚才说我叫什么名字？")
        print(f"AI回复: {response}")
        
        # 获取记忆统计
        print("\n=== 记忆系统统计 ===")
        stats = agent.get_memory_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")
            
    finally:
        # 关闭Agent
        agent.close()


if __name__ == "__main__":
    main()