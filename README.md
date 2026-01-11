# LLM Agent Memory research

## 项目简介 

这是一个用于研究和评估基于 LLM 的记忆（Memory）系统的仓库，包含：

- 一个轻量级的 Memory Agent 实现（位于 `memagent/`），用于插入、检索和更新基于向量和图谱的记忆。
- 数据生成脚本（`data_generation/`）和评测（`evaluation.py`）。



## 目录结构简要说明 

- `memagent/`：Memory Agent 的核心实现（配置、状态、图数据库、节点处理等）
- `data_generation/`：生成测试用数据的脚本（easy/ex/hard）
- `chroma_db/`：示例的 Chroma 向量数据库存放位置
- `evaluation.py`：一个使用 `MemoryAgent` 的端到端评测示例脚本

---

## 环境与依赖安装 

建议使用 Python 3.10+。在 Windows 上的示例：

1. 创建并激活虚拟环境：

```powershell
conda create --name memagent python=3.10
conda activate memagent
```

2. 安装根目录依赖：

```powershell
pip install -r requirements.txt
```

## 快速运行示例 

1. 配置API KEY
在 `memagent/config.py`, `data_generation/data_gen_easy.py`, 
`data_generation/data_gen_ex.py`, `data_generation/data_gen_hard.py`, `evaluation.py`中修改API KEY


2. 生成示例数据：

```powershell
python data_generation\data_gen_easy.py
python data_generation\data_gen_ex.py
python data_generation\data_gen_hard.py
```
将生成文件按需复制到项目根目录，并重命名为 `experiment_dataset.json`

3. 运行测试
```powershell
python evaluation.py
```




