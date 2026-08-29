"""
app/embeddings/factory.py
"""
from functools import lru_cache

from app.core.config import get_settings
from app.embeddings.base import EmbedderBase


@lru_cache(maxsize=1)
def get_embedder() -> EmbedderBase:
    """Zero-argument cached factory for the embedder. Strictly local HuggingFace."""
    settings = get_settings()
    if settings.embedding_provider == "huggingface":
        from app.embeddings.hf_embedder import HuggingFaceEmbedder
        return HuggingFaceEmbedder(model_name=settings.hf_embedding_model)
    else:
        raise ValueError(f"Unsupported embedding provider: {settings.embedding_provider}. Only 'huggingface' is active.")
