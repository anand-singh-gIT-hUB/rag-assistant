"""
app/retrieval/pipeline.py
──────────────────────────
Retrieval pipeline: embed → retrieve → filter → (optionally rerank) → top-N.
The reranker is only instantiated when reranker_enabled=True in settings,
avoiding unnecessary model loading on startup.
"""
import time
from typing import Any

from app.core.config import Settings
from app.core.logging import get_logger
from app.embeddings.base import EmbedderBase
from app.retrieval.filters import build_doc_filter
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
        # Only load the heavy CrossEncoder model if reranking is actually enabled
        self._reranker = (
            CrossEncoderReranker(model_name=settings.reranker_model)
            if settings.reranker_enabled
            else None
        )
        self._settings = settings

    def run(
        self,
        query: str,
        top_k: int | None = None,
        doc_ids: list[str] | None = None,
        rerank: bool | None = None,
    ) -> tuple[list[dict[str, Any]], bool]:
        """
        Returns (chunks, reranked_flag).
        chunks — ordered list of result dicts ready for LLM context assembly.
        """
        effective_top_k = top_k or self._settings.retrieval_top_k
        effective_rerank = rerank if rerank is not None else self._settings.reranker_enabled

        logger.info(
            "Retrieval config resolved",
            top_k_arg=top_k,
            rerank_arg=rerank,
            settings_top_k=self._settings.retrieval_top_k,
            settings_reranker_enabled=self._settings.reranker_enabled,
            effective_top_k=effective_top_k,
            effective_rerank=effective_rerank,
        )

        where = build_doc_filter(doc_ids)

        t0 = time.perf_counter()
        candidates = self._retriever.retrieve(
            query=query, top_k=effective_top_k, where=where
        )
        dense_time = time.perf_counter() - t0
        logger.info("Dense retrieval done", n=len(candidates), dense_s=round(dense_time, 2))

        if effective_rerank and self._reranker and len(candidates) > 0:
            t1 = time.perf_counter()
            chunks = self._reranker.rerank(
                query=query,
                candidates=candidates,
                top_n=self._settings.rerank_top_n,
            )
            rerank_time = time.perf_counter() - t1
            logger.info("Reranking done", n=len(chunks), rerank_s=round(rerank_time, 2))
            return chunks, True

        return candidates[: self._settings.rerank_top_n], False
