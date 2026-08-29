"""
app/retrieval/pipeline.py
──────────────────────────
Retrieval pipeline: embed → retrieve → (BM25 + RRF if hybrid) → (optionally rerank) → top-N.

Hybrid mode (hybrid_enabled=True in settings):
  1. Dense retriever   → top-K semantic candidates
  2. BM25 retriever    → top-K keyword candidates
  3. RRF fusion        → single merged, deduplicated ranking
  4. Cross-encoder reranker (optional) → final top-N

Dense-only mode (hybrid_enabled=False):
  Behavior is identical to the previous pipeline.

The BM25 index is built lazily on the first retrieve() call and
is invalidated whenever DocumentService ingests or deletes a document.
"""
import time
from typing import Any

from app.core.config import Settings
from app.core.logging import get_logger
from app.embeddings.base import EmbedderBase
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.filters import build_doc_filter
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.retriever import DenseRetriever
from app.vectorstore.base import VectorStoreBase

logger = get_logger(__name__)


class RetrievalPipeline:
    def __init__(
        self,
        embedder: EmbedderBase,
        vector_store: VectorStoreBase,
        settings: Settings,
    ) -> None:
        self._retriever = DenseRetriever(embedder=embedder, vector_store=vector_store)
        self._bm25 = BM25Retriever(vector_store=vector_store)
        # Only load the heavy CrossEncoder model if reranking is actually enabled
        self._reranker = (
            CrossEncoderReranker(model_name=settings.reranker_model)
            if settings.reranker_enabled
            else None
        )
        self._settings = settings

    # ── BM25 index invalidation ───────────────────────────────────────────────

    def invalidate_bm25(self) -> None:
        """Discard the BM25 index so it rebuilds on the next query.
        Call this after any document ingest or delete operation.
        """
        self._bm25.invalidate()

    # ── Main retrieval run ────────────────────────────────────────────────────

    def run(
        self,
        query: str,
        top_k: int | None = None,
        doc_ids: list[str] | None = None,
        rerank: bool | None = None,
    ) -> tuple[list[dict[str, Any]], bool]:
        """
        Execute the full retrieval pipeline.

        Returns:
            (chunks, reranked_flag)
            chunks — ordered list of result dicts ready for LLM context assembly.
            reranked_flag — True if cross-encoder reranking was applied.
        """
        effective_top_k   = top_k or self._settings.retrieval_top_k
        effective_rerank  = rerank if rerank is not None else self._settings.reranker_enabled
        effective_hybrid  = self._settings.hybrid_enabled

        logger.info(
            "Retrieval config resolved",
            top_k=effective_top_k,
            rerank=effective_rerank,
            hybrid=effective_hybrid,
        )

        where = build_doc_filter(doc_ids)

        # ── Dense retrieval (always runs) ─────────────────────────────────────
        t0 = time.perf_counter()
        dense_candidates = self._retriever.retrieve(
            query=query, top_k=effective_top_k, where=where
        )
        dense_time = time.perf_counter() - t0
        logger.info("Dense retrieval done", n=len(dense_candidates), dense_s=round(dense_time, 3))

        # ── Sparse BM25 retrieval + RRF fusion (hybrid mode only) ─────────────
        if effective_hybrid:
            t1 = time.perf_counter()
            sparse_candidates = self._bm25.retrieve(
                query=query, top_k=self._settings.bm25_top_k, where=where
            )
            bm25_time = time.perf_counter() - t1
            logger.info("BM25 retrieval done", n=len(sparse_candidates), bm25_s=round(bm25_time, 3))

            t2 = time.perf_counter()
            candidates = reciprocal_rank_fusion(dense_candidates, sparse_candidates)
            rrf_time = time.perf_counter() - t2
            logger.info(
                "RRF fusion done",
                n_dense=len(dense_candidates),
                n_sparse=len(sparse_candidates),
                n_fused=len(candidates),
                rrf_s=round(rrf_time, 4),
            )
        else:
            candidates = dense_candidates

        # ── Optional cross-encoder reranker ───────────────────────────────────
        if effective_rerank and self._reranker and len(candidates) > 0:
            t3 = time.perf_counter()
            chunks = self._reranker.rerank(
                query=query,
                candidates=candidates,
                top_n=self._settings.rerank_top_n,
            )
            rerank_time = time.perf_counter() - t3
            logger.info("Reranking done", n=len(chunks), rerank_s=round(rerank_time, 3))
            return chunks, True

        return candidates[: self._settings.rerank_top_n], False
