from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "PaperMem Copilot API"
    environment: str = "development"
    cors_allow_origins: str = "*"

    postgres_host: str = "127.0.0.1"
    postgres_port: int = 5432
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"
    postgres_database: str = "papermem"
    postgres_pool_size: int = 10
    postgres_max_overflow: int = 20
    postgres_echo: bool = False

    neo4j_uri: str = "bolt://127.0.0.1:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "papermemneo4j"
    neo4j_required: bool = False

    llm_api_key: str = ""
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "openai/gpt-4o-mini"

    embedding_api_key: str = ""
    embedding_base_url: str = "https://api.openai.com/v1"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 256
    embedding_batch_size: int = 96

    retrieval_top_k: int = 6
    retrieval_lexical_top_k: int = 10
    retrieval_graph_unit_limit: int = 8
    retrieval_adjacent_chunk_radius: int = 1
    answer_context_char_budget: int = 14000
    answer_evidence_chars_per_unit: int = 1600
    graph_expansion_hops: int = 2
    chunk_size: int = 1600
    chunk_overlap: int = 150
    semantic_zoom_default: str = "macro"
    semantic_llm_enabled: bool = True
    semantic_llm_file_sample_ratio: float = 0.20
    semantic_llm_file_sample_min: int = 8
    semantic_llm_file_sample_max: int = 0
    semantic_llm_selection_strategy: str = "mmr"
    semantic_llm_conversation_sample_limit: int = 4
    semantic_llm_timeout_seconds: float = 8.0
    semantic_llm_concurrency: int = 4
    relation_link_top_k: int = 2

    class Config:
        env_file = (".env", "../.env")
        env_prefix = ""


settings = Settings()
