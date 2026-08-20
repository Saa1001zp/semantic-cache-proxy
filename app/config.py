"""
конфиг прокси - все через env, чтобы в докере не париться
"""
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # --- upstream ---
    upstream_api_url: str = Field(default="", env="UPSTREAM_API_URL")
    upstream_api_key: str = Field(default="", env="UPSTREAM_API_KEY")
    upstream_model: str = Field(default="gpt-4o-mini", env="UPSTREAM_MODEL")

    # --- cache ---
    similarity_threshold: float = Field(default=0.93, env="SIMILARITY_THRESHOLD")
    cache_ttl_seconds: int = Field(default=604800, env="CACHE_TTL_SECONDS")
    max_cache_size: int = Field(default=10000, env="MAX_CACHE_SIZE")

    vector_backend: str = Field(default="memory", env="VECTOR_BACKEND")  # memory | pgvector | redis
    embedding_model: str = Field(default="all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    embedding_device: str = Field(default="cpu", env="EMBEDDING_DEVICE")
    use_onnx: bool = Field(default=False, env="USE_ONNX")

    # --- pricing ---
    price_input_per_1k: float = Field(default=0.005, env="PRICE_INPUT_PER_1K")
    price_output_per_1k: float = Field(default=0.015, env="PRICE_OUTPUT_PER_1K")

    # --- db ---
    database_url: str = Field(default="postgresql+asyncpg://postgres:postgres@localhost:5432/semantic_cache", env="DATABASE_URL")
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")

    # --- app ---
    app_port: int = Field(default=8000, env="APP_PORT")
    log_level: str = Field(default="info", env="LOG_LEVEL")

    # avg tokens per request - грубая оценка для дашборда если upstream не вернул usage
    avg_input_tokens: int = 150
    avg_output_tokens: int = 250

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
