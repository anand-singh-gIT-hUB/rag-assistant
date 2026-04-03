"""
tests/integration/test_query_pipeline.py
─────────────────────────────────────────
Integration test: runs retrieve → rerank (mocked) → answer flow.
"""
import pytest
from unittest.mock import MagicMock


def test_retrieval_pipeline_no_rerank():
    from app.retrieval.pipeline import RetrievalPipeline

    settings = MagicMock()
    settings.retrieval_top_k = 5
    settings.rerank_top_n = 3
    settings.reranker_enabled = False
    settings.reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    embedder = MagicMock()
    embedder.embed_query.return_value = [0.0] * 10

    store = MagicMock()
    store.query.return_value = [
        {"chunk_id": f"c{i}", "text": f"chunk {i}", "metadata": {}, "score": 1.0 - i * 0.1}
        for i in range(5)
    ]

    pipeline = RetrievalPipeline(embedder=embedder, vector_store=store, settings=settings)
    results, reranked = pipeline.run(query="test", rerank=False)

    assert not reranked
    assert len(results) <= 3
