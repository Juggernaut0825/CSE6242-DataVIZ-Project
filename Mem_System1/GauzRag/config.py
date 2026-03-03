"""
GauzRag 配置管理模块
"""
import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field


def load_env_file(env_path: Path) -> None:
    """加载 .env 文件到环境变量"""
    if not env_path.exists():
        return
    
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()


@dataclass
class GauzRagConfig:
    """GauzRag 系统配置"""
    
    # 项目路径
    project_root: Path = field(default_factory=lambda: Path.cwd())
    
    # LLM 配置
    llm_provider: str = "openai"
    llm_api_base: str = "https://openrouter.ai/api/v1"
    llm_api_key: str = ""
    llm_model: str = "openai/gpt-4o-mini"
    llm_temperature: float = 0.3
    llm_max_tokens: Optional[int] = None
    
    # Embedding 配置
    embedding_api_key: str = ""
    embedding_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embedding_model: str = "text-embedding-v4"
    
    # MySQL 配置
    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = ""
    mysql_database: str = "gauzrag"
    mysql_table: str = "facts"
    
    # Qdrant 配置
    qdrant_mode: str = "server"  # "server" 或 "local"
    qdrant_url: str = "http://localhost:6333"  # Server 模式的 URL
    
    # 输出路径
    output_dir: Path = field(default_factory=lambda: Path("output"))
    cache_dir: Path = field(default_factory=lambda: Path("cache"))
    logs_dir: Path = field(default_factory=lambda: Path("logs"))
    
    # 图谱引擎配置
    graph_engine: str = "lightrag"  # 图谱引擎（固定使用 LightRAG）
    
    # Neo4j 配置（用于 LightRAG）
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    
    # LightRAG 配置
    lightrag_kv_storage: str = "JsonKVStorage"
    lightrag_vector_storage: str = "NanoVectorDBStorage"
    lightrag_graph_storage: str = "NetworkXStorage"
    lightrag_chunk_token_size: int = 1200
    lightrag_chunk_overlap_token_size: int = 100
    lightrag_entity_extract_max_gleaning: int = 1
    
    # GraphRAG 配置
    graphrag_workflows: list = field(default_factory=lambda: [
        "load_input_documents",
        "create_base_text_units",
        "create_final_documents",
        "extract_graph",
        "finalize_graph",
        "extract_covariates",
        "create_communities",
        "create_final_text_units",
        "create_community_reports",
    ])
    
    # 召回配置
    search_top_k: int = 3
    dedupe_threshold: float = 0.85
    max_facts_per_community: int = 10
    
    # 社区检测配置
    max_cluster_size: int = 100         # GraphRAG 社区最大尺寸（增大以提高覆盖率）
    use_lcc: bool = False               # 是否只使用最大连通分量
    cluster_seed: int = 0xDEADBEEF     # 聚类随机种子
    min_community_size: int = 1         # 社区最小尺寸（允许单节点社区）
    
    # 实体提取配置
    entity_types: list = field(default_factory=lambda: [
        "organization", "person", "geo", "event",  # 原有类型
        "technology", "model", "algorithm", "concept",  # 技术类
        "method", "technique", "system", "tool"  # 方法类
    ])
    max_gleanings: int = 1              # LLM 提取轮数（增加到1次补充提取，提高召回率）
    concurrent_requests: int = 25       # 并发请求数（独立模式下并行处理facts）
    requests_per_minute: int = 300      # 每分钟最大请求数
    
    @classmethod
    def from_env(cls, env_path: Optional[Path] = None) -> "GauzRagConfig":
        """从环境变量加载配置"""
        if env_path:
            load_env_file(env_path)
        
        # 解析 max_tokens
        max_tokens_str = os.getenv("GAUZ_LLM_MAX_TOKENS", "").strip().lower()
        max_tokens = None
        if max_tokens_str and max_tokens_str not in ["none", "unlimited", "0"]:
            try:
                max_tokens = int(max_tokens_str)
            except ValueError:
                max_tokens = None
        
        config = cls(
            llm_provider=os.getenv("GAUZ_LLM_PROVIDER", "openai"),
            llm_api_base=os.getenv("GAUZ_LLM_API_BASE", "https://openrouter.ai/api/v1"),
            llm_api_key=os.getenv("GAUZ_LLM_API_KEY", ""),
            llm_model=os.getenv("GAUZ_LLM_MODEL", "openai/gpt-4o-mini"),
            llm_temperature=float(os.getenv("GAUZ_LLM_TEMPERATURE", "0.3")),
            llm_max_tokens=max_tokens,
            
            embedding_api_key=os.getenv("DASHSCOPE_API_KEY", ""),
            embedding_base_url=os.getenv(
                "DASHSCOPE_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1"
            ),
            embedding_model=os.getenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v4"),
            
            mysql_host=os.getenv("MYSQL_HOST", "127.0.0.1"),
            mysql_port=int(os.getenv("MYSQL_PORT", "3306")),
            mysql_user=os.getenv("MYSQL_USER", "root"),
            mysql_password=os.getenv("MYSQL_PASSWORD", ""),
            mysql_database=os.getenv("MYSQL_DB", "gauzrag"),
            mysql_table=os.getenv("MYSQL_TABLE", "facts"),
            
            qdrant_mode=os.getenv("QDRANT_MODE", "server"),
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            
            # 图谱引擎选择
            graph_engine="lightrag",  # 固定使用 LightRAG
            
            # Neo4j 配置
            neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
            neo4j_password=os.getenv("NEO4J_PASSWORD", ""),
            
            # LightRAG 配置
            lightrag_kv_storage=os.getenv("LIGHTRAG_KV_STORAGE", "JsonKVStorage"),
            lightrag_vector_storage=os.getenv("LIGHTRAG_VECTOR_STORAGE", "NanoVectorDBStorage"),
            lightrag_graph_storage=os.getenv("LIGHTRAG_GRAPH_STORAGE", "NetworkXStorage"),
            lightrag_chunk_token_size=int(os.getenv("LIGHTRAG_CHUNK_TOKEN_SIZE", "1200")),
            lightrag_chunk_overlap_token_size=int(os.getenv("LIGHTRAG_CHUNK_OVERLAP_TOKEN_SIZE", "100")),
            lightrag_entity_extract_max_gleaning=int(os.getenv("LIGHTRAG_ENTITY_EXTRACT_MAX_GLEANING", "1")),
            
            max_cluster_size=int(os.getenv("GAUZ_MAX_CLUSTER_SIZE", "100")),
            min_community_size=int(os.getenv("GAUZ_MIN_COMMUNITY_SIZE", "2")),
            use_lcc=os.getenv("GAUZ_USE_LCC", "false").lower() in ("true", "1", "yes"),
        )
        
        return config
    
    def setup_directories(self) -> None:
        """创建必要的目录"""
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.logs_dir.mkdir(exist_ok=True, parents=True)
        (self.project_root / "input").mkdir(exist_ok=True, parents=True)
    
    def validate(self) -> bool:
        """验证配置是否完整"""
        errors = []
        
        if not self.llm_api_key:
            errors.append("未配置 GAUZ_LLM_API_KEY")
        
        if not self.embedding_api_key:
            errors.append("未配置 DASHSCOPE_API_KEY")
        
        if not self.mysql_password:
            errors.append("未配置 MYSQL_PASSWORD")
        
        if errors:
            print("配置错误:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True

