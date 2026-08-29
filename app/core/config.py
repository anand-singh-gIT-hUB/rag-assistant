"""
app/core/config.py
──────────────────
Centralised settings loaded from environment variables / .env file.
All other modules import `get_settings()` — never os.getenv() directly.
"""
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Environment ────────────────────────────────────────────────────────────
    app_env: Literal["dev", "prod"] = "dev"
    enable_evaluation: bool = True  # Defaulted in constructor
 
    # ── LLM ──────────────────────────────────────────────────────────────────
    llm_provider: Literal["ollama"] = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"

    # ── Embeddings ───────────────────────────────────────────────────────────
    embedding_provider: Literal["huggingface"] = "huggingface"
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
    reranker_enabled: bool = True  # Defaulted in constructor
    # ── Hybrid Retrieval (BM25 + Dense) ──────────────────────────────────────
    hybrid_enabled: bool = True    # Toggle BM25 + dense fusion
    bm25_top_k: int = 20           # BM25 candidate pool size before RRF

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


def get_settings() -> Settings:
    """Return the cached Settings singleton."""
    s = Settings()
    # Apply environment-aware defaults if not explicitly set in .env/environment
    # We do this manually here because Pydantic v2's default-setting-based-on-other-fields
    # is cleanest via model_validator or post-init logic.
    if "RERANKER_ENABLED" not in os.environ and s.app_env == "prod":
        object.__setattr__(s, "reranker_enabled", False)
    if "ENABLE_EVALUATION" not in os.environ and s.app_env == "prod":
        object.__setattr__(s, "enable_evaluation", False)
    return s
