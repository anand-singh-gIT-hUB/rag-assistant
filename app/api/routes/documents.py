"""
app/api/routes/documents.py
────────────────────────────
Endpoints: upload, list, delete, reindex documents.
After each mutation (upload, delete, reindex) the BM25 index is invalidated
so it rebuilds fresh on the next hybrid retrieval query.
"""
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse

from app.api.dependencies import get_document_service, get_retrieval_pipeline
from app.core.exceptions import (
    DocumentNotFoundError,
    UnsupportedFileTypeError,
    DocumentProcessingError,
)
from app.retrieval.pipeline import RetrievalPipeline
from app.schemas.document import DocumentListResponse, DocumentResponse
from app.services.document_service import DocumentService

router = APIRouter()


@router.post(
    "/upload",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and index a document",
)
async def upload_document(
    file: UploadFile = File(...),
    service: DocumentService = Depends(get_document_service),
    pipeline: RetrievalPipeline = Depends(get_retrieval_pipeline),
) -> DocumentResponse:
    try:
        result = await service.ingest(file)
        pipeline.invalidate_bm25()   # BM25 index must rebuild after new doc
        return result
    except UnsupportedFileTypeError as e:
        raise HTTPException(status_code=415, detail=e.detail)
    except DocumentProcessingError as e:
        raise HTTPException(status_code=422, detail=e.detail)


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List all indexed documents",
)
async def list_documents(
    service: DocumentService = Depends(get_document_service),
) -> DocumentListResponse:
    return await service.list_documents()


@router.delete(
    "/{doc_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete document and its vectors",
)
async def delete_document(
    doc_id: str,
    service: DocumentService = Depends(get_document_service),
    pipeline: RetrievalPipeline = Depends(get_retrieval_pipeline),
) -> None:
    try:
        await service.delete(doc_id)
        pipeline.invalidate_bm25()   # BM25 index must rebuild after deletion
    except DocumentNotFoundError as e:
        raise HTTPException(status_code=404, detail=e.detail)


@router.post(
    "/{doc_id}/reindex",
    response_model=DocumentResponse,
    summary="Re-parse and re-embed a document",
)
async def reindex_document(
    doc_id: str,
    service: DocumentService = Depends(get_document_service),
    pipeline: RetrievalPipeline = Depends(get_retrieval_pipeline),
) -> DocumentResponse:
    try:
        result = await service.reindex(doc_id)
        pipeline.invalidate_bm25()   # BM25 index must rebuild after reindex
        return result
    except DocumentNotFoundError as e:
        raise HTTPException(status_code=404, detail=e.detail)
    except DocumentProcessingError as e:
        raise HTTPException(status_code=422, detail=e.detail)
