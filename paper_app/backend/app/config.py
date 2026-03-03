from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "PaperMem Copilot API"
    environment: str = "development"

    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = "password"
    mysql_database: str = "papermem"
    mysql_charset: str = "utf8mb4"
    mysql_pool_size: int = 10
    mysql_max_overflow: int = 20

    memory_api_base: str = "http://43.139.19.144:1235"
    neo4j_uri: str = "bolt://43.139.19.144:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    gemini_model: str = "google/gemini-3-flash-preview"
    cors_allow_origins: str = "*"

    class Config:
        env_file = (".env", "../.env")
        env_prefix = ""


settings = Settings()
