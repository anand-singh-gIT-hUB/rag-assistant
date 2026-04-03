"""
app/api/dependencies.py
───────────────────────
FastAPI dependency-injection helpers shared across routers.
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


def get_query_service(
    settings: Settings = Depends(get_settings),
) -> QueryService:
    embedder = get_embedder(settings)
    store = get_vector_store(settings)
    llm = get_llm(settings)
    pipeline = RetrievalPipeline(embedder=embedder, vector_store=store, settings=settings)
    return QueryService(retrieval_pipeline=pipeline, llm=llm, settings=settings)


def get_evaluation_service(
    settings: Settings = Depends(get_settings),
) -> EvaluationService:
    return EvaluationService(settings=settings)
