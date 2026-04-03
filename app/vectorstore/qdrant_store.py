"""
app/vectorstore/qdrant_store.py
────────────────────────────────
Qdrant vector store stub — implement when switching providers.
"""
from typing import Any

from app.vectorstore.base import VectorStoreBase
from app.processing.metadata import ChunkMetadata


class QdrantStore(VectorStoreBase):
    """Stub implementation — replace with qdrant-client calls."""

    def __init__(self, url: str, collection_name: str) -> None:
        self._url = url
        self._collection_name = collection_name
        # from qdrant_client import QdrantClient
        # self._client = QdrantClient(url=url)

    def add_chunks(self, chunks: list[ChunkMetadata], embeddings: list[list[float]]) -> None:
        raise NotImplementedError("QdrantStore.add_chunks not yet implemented")

    def query(self, embedding: list[float], top_k: int = 10, where: dict | None = None) -> list[dict[str, Any]]:
        raise NotImplementedError("QdrantStore.query not yet implemented")

    def delete_by_doc_id(self, doc_id: str) -> None:
        raise NotImplementedError("QdrantStore.delete_by_doc_id not yet implemented")

    def list_documents(self) -> list[dict[str, Any]]:
        raise NotImplementedError("QdrantStore.list_documents not yet implemented")

    def count(self) -> int:
        raise NotImplementedError("QdrantStore.count not yet implemented")
