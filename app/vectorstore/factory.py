"""
app/vectorstore/factory.py
"""
from functools import lru_cache

from app.core.config import Settings
from app.vectorstore.base import VectorStoreBase


@lru_cache(maxsize=1)
def get_vector_store(settings: Settings) -> VectorStoreBase:
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
