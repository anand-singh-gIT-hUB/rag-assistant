"""
app/embeddings/factory.py
"""
from functools import lru_cache

from typing import Optional

from app.core.config import Settings, get_settings
from app.embeddings.base import EmbedderBase


@lru_cache(maxsize=1)
def _get_embedder() -> EmbedderBase:
    settings = get_settings()
    if settings.embedding_provider == "openai":
        from app.embeddings.openai_embedder import OpenAIEmbedder
        return OpenAIEmbedder(
            api_key=settings.openai_api_key,
            model=settings.openai_embedding_model,
        )
    elif settings.embedding_provider == "huggingface":
        from app.embeddings.hf_embedder import HuggingFaceEmbedder
        return HuggingFaceEmbedder(model_name=settings.hf_embedding_model)
    else:
        raise ValueError(f"Unknown embedding provider: {settings.embedding_provider}")


def get_embedder(settings: Optional[Settings] = None) -> EmbedderBase:
    """Public wrapper to get the cached embedder."""
    return _get_embedder()
