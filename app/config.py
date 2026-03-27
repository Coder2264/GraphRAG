"""Application settings — all values sourced from .env file."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # API Keys
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # Gemini
    gemini_api_key: str = ""
    gemini_extraction_model: str = "gemini-3-flash"
    gemini_llm_model: str = "gemini-3-flash"

    # Databases
    postgres_dsn: str = ""
    neo4j_uri: str = ""
    neo4j_user: str = ""
    neo4j_password: str = ""
    neo4j_database: str = "graphRAG"

    # Ollama
    ollama_base_url: str = ""
    ollama_llm_model: str = ""
    ollama_embed_model: str = ""
    ollama_extraction_timeout: float = 600.0  # seconds; large docs need >2 min

    # Embedding
    embedding_dim: int = 0

    # Chunking (RAG vector store)
    chunk_size: int = 0
    chunk_overlap: int = 0

    # Graph extraction chunking (separate from RAG — larger chunks for entity context)
    graph_extraction_chunk_size: int = 6000
    graph_extraction_chunk_overlap: int = 800

    # Default providers
    default_llm: str = ""
    default_embedder: str = ""
    default_graph_store: str = ""
    default_vector_store: str = ""
    default_entity_extractor: str = ""

    # GraphRAG beam-search retriever
    beam_search_max_iterations: int = 20
    beam_search_beam_width: int = 8

    # Think-on-Graph (ToG / ToG-R) retriever
    tog_depth_max: int = 3


settings = Settings()
