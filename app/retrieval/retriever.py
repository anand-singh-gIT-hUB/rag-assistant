"""
app/retrieval/retriever.py
───────────────────────────
Dense retrieval: embed query → query vector store → return top-k results.
"""
from typing import Any

from app.embeddings.base import EmbedderBase
from app.vectorstore.base import VectorStoreBase


class DenseRetriever:
    def __init__(self, embedder: EmbedderBase, vector_store: VectorStoreBase) -> None:
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        where: dict | None = None,
    ) -> list[dict[str, Any]]:
        embedding = self.embedder.embed_query(query)
        return self.vector_store.query(embedding=embedding, top_k=top_k, where=where)
