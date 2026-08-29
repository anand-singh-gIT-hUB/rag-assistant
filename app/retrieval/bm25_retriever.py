"""
app/retrieval/bm25_retriever.py
───────────────────────────────
Sparse BM25 retriever. Builds an in-memory BM25Okapi index from the full
text corpus stored in the vector store and returns ranked candidates.

The index is built lazily on the first retrieve() call and can be
invalidated (e.g., after a document ingest or delete) so it rebuilds
automatically on the next request.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from app.core.logging import get_logger

if TYPE_CHECKING:
    from app.vectorstore.base import VectorStoreBase

logger = get_logger(__name__)

_TOKENIZE_RE = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase word-tokenizer (no NLTK required)."""
    return _TOKENIZE_RE.findall(text.lower())


class BM25Retriever:
    """
    In-memory BM25 sparse retriever backed by rank_bm25.BM25Okapi.

    Usage:
        retriever = BM25Retriever(vector_store)
        results   = retriever.retrieve(query="what is EcoTwin?", top_k=20)
        retriever.invalidate()   # called after document ingest/delete
    """

    def __init__(self, vector_store: VectorStoreBase) -> None:
        self._vector_store = vector_store
        self._corpus: list[dict[str, Any]] = []   # [{chunk_id, text, metadata}, ...]
        self._bm25 = None                          # BM25Okapi | None
        self._is_built = False

    # ── Index lifecycle ───────────────────────────────────────────────────────

    def _build(self) -> None:
        """Fetch all chunks from the vector store and fit a BM25 index."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as e:
            logger.warning(
                "rank-bm25 is not installed. Hybrid retrieval will fall back to dense-only mode.",
                error=str(e),
            )
            self._bm25 = None
            self._is_built = True
            return

        logger.info("Building BM25 index from vector store corpus…")
        self._corpus = self._vector_store.get_all_chunks()

        if not self._corpus:
            logger.warning("BM25 corpus is empty — skipping index build.")
            self._bm25 = None
            self._is_built = True
            return

        tokenized = [_tokenize(c["text"]) for c in self._corpus]
        self._bm25 = BM25Okapi(tokenized)
        self._is_built = True
        logger.info("BM25 index built", n_chunks=len(self._corpus))

    def invalidate(self) -> None:
        """Discard the index so it is rebuilt on the next retrieve() call."""
        self._is_built = False
        self._bm25 = None
        self._corpus = []
        logger.info("BM25 index invalidated — will rebuild on next query.")

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        where: dict | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return the top-k BM25-ranked chunks.

        Each result dict has keys: chunk_id, text, metadata, score.
        `where` filters by doc_id (same semantics as ChromaDB where-filter).
        """
        if not self._is_built:
            self._build()

        if not self._corpus or self._bm25 is None:
            return []

        tokens = _tokenize(query)
        raw_scores: list[float] = self._bm25.get_scores(tokens).tolist()

        # Apply doc_id filter if requested (mirrors ChromaDB `where` semantics)
        allowed_ids: set[str] | None = None
        if where:
            if "$eq" in str(where):
                val = list(where.values())[0]
                if isinstance(val, dict):
                    allowed_ids = {val.get("$eq", "")}
                else:
                    allowed_ids = {str(val)}
            elif "$in" in str(where):
                val = list(where.values())[0]
                if isinstance(val, dict):
                    allowed_ids = set(val.get("$in", []))

        scored: list[tuple[float, dict]] = []
        for score, chunk in zip(raw_scores, self._corpus):
            if allowed_ids and chunk["metadata"].get("doc_id") not in allowed_ids:
                continue
            scored.append((score, chunk))

        # Sort descending and take top-k
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k]

        # Normalise scores to [0, 1] range
        max_score = top[0][0] if top and top[0][0] > 0 else 1.0
        results = []
        for raw, chunk in top:
            results.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "metadata": chunk["metadata"],
                    "score": raw / max_score,   # normalised BM25 score
                }
            )

        return results
