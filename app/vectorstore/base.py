"""
app/vectorstore/base.py
"""
from abc import ABC, abstractmethod
from typing import Any

from app.processing.metadata import ChunkMetadata


class VectorStoreBase(ABC):

    @abstractmethod
    def add_chunks(
        self, chunks: list[ChunkMetadata], embeddings: list[list[float]]
    ) -> None:
        ...

    @abstractmethod
    def query(
        self,
        embedding: list[float],
        top_k: int = 10,
        where: dict | None = None,
    ) -> list[dict[str, Any]]:
        """Return list of dicts with keys: chunk_id, text, metadata, score."""
        ...

    @abstractmethod
    def delete_by_doc_id(self, doc_id: str) -> None:
        ...

    @abstractmethod
    def list_documents(self) -> list[dict[str, Any]]:
        """Return unique document metadata records."""
        ...

    @abstractmethod
    def count(self) -> int:
        ...
