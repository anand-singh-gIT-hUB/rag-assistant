"""
app/vectorstore/chroma_store.py
────────────────────────────────
ChromaDB vector store implementation.
"""
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.exceptions import VectorStoreError
from app.processing.metadata import ChunkMetadata
from app.vectorstore.base import VectorStoreBase


class ChromaStore(VectorStoreBase):
    def __init__(self, persist_dir: str, collection_name: str) -> None:
        try:
            self._client = chromadb.PersistentClient(
                path=persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._col = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as e:
            raise VectorStoreError(f"ChromaDB init failed: {e}") from e

    def add_chunks(
        self, chunks: list[ChunkMetadata], embeddings: list[list[float]]
    ) -> None:
        if not chunks:
            return
        try:
            self._col.upsert(
                ids=[c.chunk_id for c in chunks],
                embeddings=embeddings,
                documents=[c.text for c in chunks],
                metadatas=[c.to_chroma_metadata() for c in chunks],
            )
        except Exception as e:
            raise VectorStoreError(f"ChromaDB upsert failed: {e}") from e

    def query(
        self,
        embedding: list[float],
        top_k: int = 10,
        where: dict | None = None,
    ) -> list[dict[str, Any]]:
        try:
            kwargs: dict = dict(
                query_embeddings=[embedding],
                n_results=min(top_k, self._col.count() or 1),
                include=["documents", "metadatas", "distances"],
            )
            if where:
                kwargs["where"] = where
            result = self._col.query(**kwargs)
            out = []
            for i, chunk_id in enumerate(result["ids"][0]):
                out.append(
                    {
                        "chunk_id": chunk_id,
                        "text": result["documents"][0][i],
                        "metadata": result["metadatas"][0][i],
                        "score": 1.0 - result["distances"][0][i],  # cosine → similarity
                    }
                )
            return out
        except Exception as e:
            raise VectorStoreError(f"ChromaDB query failed: {e}") from e

    def delete_by_doc_id(self, doc_id: str) -> None:
        try:
            self._col.delete(where={"doc_id": doc_id})
        except Exception as e:
            raise VectorStoreError(f"ChromaDB delete failed: {e}") from e

    def list_documents(self) -> list[dict[str, Any]]:
        try:
            result = self._col.get(include=["metadatas"])
            seen: dict[str, dict] = {}
            for meta in result["metadatas"]:
                doc_id = meta.get("doc_id", "")
                if doc_id not in seen:
                    seen[doc_id] = meta
            return list(seen.values())
        except Exception as e:
            raise VectorStoreError(f"ChromaDB list failed: {e}") from e

    def count(self) -> int:
        return self._col.count()
