"""
tests/unit/test_query_service.py
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.services.query_service import QueryService
from app.schemas.query import QueryRequest


@pytest.fixture
def mock_service(tmp_path):
    settings = MagicMock()
    settings.retrieval_top_k = 10
    settings.rerank_top_n = 5
    settings.reranker_enabled = False
    settings.logs_path = tmp_path

    pipeline = MagicMock()
    pipeline.run.return_value = (
        [{"chunk_id": "c1", "text": "The answer is 42.", "metadata": {"doc_id": "d1", "file_name": "a.pdf"}, "score": 0.9}],
        False,
    )

    llm = MagicMock()
    llm.model_name = "test-model"
    llm.complete.return_value = "The answer is 42."

    return QueryService(retrieval_pipeline=pipeline, llm=llm, settings=settings)


@pytest.mark.asyncio
async def test_answer_returns_response(mock_service):
    req = QueryRequest(question="What is the answer?")
    response = await mock_service.answer(req)
    assert response.answer == "The answer is 42."
    assert response.model == "test-model"
    assert len(response.citations) == 1


@pytest.mark.asyncio
async def test_answer_logs_query(mock_service, tmp_path):
    req = QueryRequest(question="Test?")
    await mock_service.answer(req)
    log_file = tmp_path / "queries.jsonl"
    assert log_file.exists()
    assert "Test?" in log_file.read_text()
