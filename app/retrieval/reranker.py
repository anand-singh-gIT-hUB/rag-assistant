"""
app/retrieval/reranker.py
──────────────────────────
Cross-encoder reranker using HuggingFace sentence-transformers.
Reranks candidate chunks by (query, passage) relevance.
"""
from typing import Any

from app.core.exceptions import EmbeddingError


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self._model_name = model_name
        self._model = None  # lazy-load

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self._model_name)
            except Exception as e:
                raise EmbeddingError(f"Failed to load CrossEncoder: {e}") from e
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_n: int = 5,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []
        model = self._get_model()
        pairs = [(query, c["text"]) for c in candidates]
        scores = model.predict(pairs).tolist()
        for chunk, score in zip(candidates, scores):
            chunk["rerank_score"] = score
        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_n]
