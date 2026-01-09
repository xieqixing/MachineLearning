class MemoryAgentConfig:
    """Agent配置类"""
    def __init__(
        self,
        agent_name: str = "MemoryAgent",
        llm_model: str = "qwen-plus",
        llm_temperature: float = 0.1,
        llm_api_key: str = "sk-2770a3f619c14f31a87d47924de34af2",
        llm_api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        embedding_model: str = "all-MiniLM-L6-v2",
        short_term_window_size: int = 4,
        summary_batch_size: int = 2,
        vector_store_path: str = "./chroma_db",
        checkpoints_db: str = "checkpoints.db",
        verbose: bool = True
    ):
        self.agent_name = agent_name
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.llm_api_key = llm_api_key
        self.llm_api_base = llm_api_base
        self.embedding_model = embedding_model
        self.short_term_window_size = short_term_window_size
        self.summary_batch_size = summary_batch_size
        self.vector_store_path = vector_store_path
        self.checkpoints_db = checkpoints_db
        self.verbose = verbose