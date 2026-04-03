"""
app/retrieval/pipeline.py
──────────────────────────
Retrieval pipeline: embed → retrieve → filter → rerank → top-N.
"""
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
        self._reranker = CrossEncoderReranker(model_name=settings.reranker_model)
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

        where = build_doc_filter(doc_ids)
        candidates = self._retriever.retrieve(
            query=query, top_k=effective_top_k, where=where
        )
        logger.info("Retrieved candidates", n=len(candidates), rerank=effective_rerank)

        if effective_rerank and len(candidates) > 0:
            chunks = self._reranker.rerank(
                query=query,
                candidates=candidates,
                top_n=self._settings.rerank_top_n,
            )
            return chunks, True

        return candidates[: self._settings.rerank_top_n], False
