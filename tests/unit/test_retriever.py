"""
tests/unit/test_retriever.py
"""
from unittest.mock import MagicMock
from app.retrieval.retriever import DenseRetriever


def test_retrieve_calls_embedder_and_store():
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.1, 0.2, 0.3]
    store = MagicMock()
    store.query.return_value = [{"chunk_id": "c1", "text": "hello", "metadata": {}, "score": 0.9}]

    retriever = DenseRetriever(embedder=embedder, vector_store=store)
    results = retriever.retrieve("test query", top_k=5)

    embedder.embed_query.assert_called_once_with("test query")
    store.query.assert_called_once_with(embedding=[0.1, 0.2, 0.3], top_k=5, where=None)
    assert len(results) == 1
    assert results[0]["chunk_id"] == "c1"
