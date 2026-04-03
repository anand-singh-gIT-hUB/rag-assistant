"""
app/core/exceptions.py
──────────────────────
Custom exception hierarchy for the RAG Assistant.
FastAPI exception handlers map these to appropriate HTTP responses.
"""
from fastapi import HTTPException, status


class RAGBaseException(Exception):
    """Root exception for all RAG Assistant errors."""

    def __init__(self, message: str, detail: str | None = None) -> None:
        self.message = message
        self.detail = detail or message
        super().__init__(self.message)


# ── Document / Ingestion ──────────────────────────────────────────────────────

class DocumentNotFoundError(RAGBaseException):
    """Raised when a requested document_id does not exist."""


class UnsupportedFileTypeError(RAGBaseException):
    """Raised when the uploaded file type is not supported."""


class DocumentProcessingError(RAGBaseException):
    """Raised when parsing or chunking a document fails."""


class StorageError(RAGBaseException):
    """Raised when a file-system operation fails."""


# ── Embedding / Vector Store ──────────────────────────────────────────────────

class EmbeddingError(RAGBaseException):
    """Raised when an embedding call fails."""


class VectorStoreError(RAGBaseException):
    """Raised when a vector-store operation fails."""


# ── LLM ───────────────────────────────────────────────────────────────────────

class LLMError(RAGBaseException):
    """Raised when an LLM completion fails."""


class ContextTooLongError(RAGBaseException):
    """Raised when assembled context exceeds the token budget."""


# ── Query ─────────────────────────────────────────────────────────────────────

class QueryValidationError(RAGBaseException):
    """Raised when query input is invalid."""


# ── Evaluation ────────────────────────────────────────────────────────────────

class EvaluationError(RAGBaseException):
    """Raised when a Ragas evaluation fails."""


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def not_found(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=detail)


def bad_request(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


def internal_error(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=detail
    )
