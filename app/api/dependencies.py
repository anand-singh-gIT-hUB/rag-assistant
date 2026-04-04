"""
app/api/dependencies.py
───────────────────────
FastAPI dependency-injection helpers shared across routers.
The QueryService is cached as a true singleton — no settings object
is passed as a cache key to avoid any hashing or invalidation issues.
"""
from functools import lru_cache

from fastapi import Depends

from app.core.config import Settings, get_settings
from app.embeddings.factory import get_embedder
from app.vectorstore.factory import get_vector_store
from app.llm.factory import get_llm
from app.retrieval.pipeline import RetrievalPipeline
from app.services.document_service import DocumentService
from app.services.query_service import QueryService
from app.services.evaluation_service import EvaluationService


def get_document_service(
    settings: Settings = Depends(get_settings),
) -> DocumentService:
    embedder = get_embedder(settings)
    store = get_vector_store(settings)
    return DocumentService(embedder=embedder, vector_store=store, settings=settings)


@lru_cache(maxsize=1)
def _build_query_service() -> QueryService:
    """Build the canonical, long-lived QueryService singleton."""
    settings = get_settings()
    embedder = get_embedder(settings)
    store = get_vector_store(settings)
    llm = get_llm(settings)
    pipeline = RetrievalPipeline(embedder=embedder, vector_store=store, settings=settings)
    return QueryService(retrieval_pipeline=pipeline, llm=llm, settings=settings)


def get_query_service() -> QueryService:
    return _build_query_service()


def get_evaluation_service() -> EvaluationService:
    settings = get_settings()
    query_service = _build_query_service()
    return EvaluationService(settings=settings, query_service=query_service)
