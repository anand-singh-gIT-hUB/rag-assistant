"""
app/core/config.py
──────────────────
Centralised settings loaded from environment variables / .env file.
All other modules import `get_settings()` — never os.getenv() directly.
"""
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        frozen=True,
    )

    # ── LLM ──────────────────────────────────────────────────────────────────
    llm_provider: Literal["openai", "ollama"] = "openai"
    openai_api_key: str = Field(default="", repr=False)
    openai_llm_model: str = "gpt-4.1-mini"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"

    # ── Embeddings ───────────────────────────────────────────────────────────
    embedding_provider: Literal["openai", "huggingface"] = "openai"
    openai_embedding_model: str = "text-embedding-3-small"
    hf_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ── Vector Store ─────────────────────────────────────────────────────────
    vector_store_provider: Literal["chroma", "qdrant"] = "chroma"
    chroma_persist_dir: str = "app/storage/vectordb"
    chroma_collection_name: str = "rag_documents"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_name: str = "rag_documents"

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieval_top_k: int = 20
    rerank_top_n: int = 5
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_enabled: bool = True

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 64

    # ── API ───────────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    allowed_origins: str = "http://localhost:8501,http://localhost:3000"

    # ── Storage ───────────────────────────────────────────────────────────────
    files_dir: str = "app/storage/files"
    logs_dir: str = "app/storage/logs"

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = "INFO"

    # ── Derived helpers ───────────────────────────────────────────────────────
    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",")]

    @property
    def files_path(self) -> Path:
        p = Path(self.files_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def logs_path(self) -> Path:
        p = Path(self.logs_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def chroma_persist_path(self) -> Path:
        p = Path(self.chroma_persist_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @field_validator("log_level")
    @classmethod
    def _upper_log_level(cls, v: str) -> str:
        return v.upper()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached Settings singleton."""
    return Settings()
