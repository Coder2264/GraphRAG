"""Application settings loaded from environment variables / .env file."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- API Keys ---
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # --- Databases ---
    postgres_dsn: str = "postgresql+asyncpg://user:password@localhost:5432/graphrag"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    # --- Default providers (looked up in registry.py) ---
    # These are the *default* keys; individual routes can override via the registry.
    default_llm: str = "in_memory"
    default_embedder: str = "in_memory"
    default_graph_store: str = "in_memory"
    default_vector_store: str = "in_memory"


settings = Settings()
