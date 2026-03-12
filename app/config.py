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

    # --- Ollama ---
    ollama_base_url: str = "http://localhost:11434"
    ollama_llm_model: str = "llama3.2"
    ollama_embed_model: str = "nomic-embed-text"

    # --- Embedding ---
    # Dimension MUST match the embedding model.
    # nomic-embed-text → 768  |  mxbai-embed-large → 1024  |  OpenAI text-embedding-3-small → 1536
    embedding_dim: int = 768

    # --- Chunking ---
    chunk_size: int = 512    # characters per chunk
    chunk_overlap: int = 64  # overlap between consecutive chunks

    # --- Default providers (looked up in registry.py) ---
    # Switch instantly by changing these values (or setting env vars).
    # Options: "ollama" | "in_memory" | "openai" | ...
    default_llm: str = "ollama"
    # Options: "ollama" | "in_memory" | "openai" | ...
    default_embedder: str = "ollama"
    # Options: "in_memory" | "neo4j"
    default_graph_store: str = "in_memory"
    # Options: "postgres" | "in_memory"
    default_vector_store: str = "postgres"


settings = Settings()
