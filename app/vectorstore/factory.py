"""
app/vectorstore/factory.py
"""
from functools import lru_cache

from typing import Optional

from app.core.config import Settings, get_settings
from app.vectorstore.base import VectorStoreBase


@lru_cache(maxsize=1)
def _get_vector_store() -> VectorStoreBase:
    settings = get_settings()
    if settings.vector_store_provider == "chroma":
        from app.vectorstore.chroma_store import ChromaStore
        return ChromaStore(
            persist_dir=str(settings.chroma_persist_path),
            collection_name=settings.chroma_collection_name,
        )
    elif settings.vector_store_provider == "qdrant":
        from app.vectorstore.qdrant_store import QdrantStore
        return QdrantStore(
            url=settings.qdrant_url,
            collection_name=settings.qdrant_collection_name,
        )
    else:
        raise ValueError(f"Unknown vector store provider: {settings.vector_store_provider}")


def get_vector_store(settings: Optional[Settings] = None) -> VectorStoreBase:
    """Public wrapper to get the cached vector store."""
    return _get_vector_store()
