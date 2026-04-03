"""
app/api/routes/documents.py
────────────────────────────
Endpoints: upload, list, delete, reindex documents.
"""
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse

from app.api.dependencies import get_document_service
from app.core.exceptions import (
    DocumentNotFoundError,
    UnsupportedFileTypeError,
    DocumentProcessingError,
)
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
) -> DocumentResponse:
    try:
        return await service.ingest(file)
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
) -> None:
    try:
        await service.delete(doc_id)
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
) -> DocumentResponse:
    try:
        return await service.reindex(doc_id)
    except DocumentNotFoundError as e:
        raise HTTPException(status_code=404, detail=e.detail)
    except DocumentProcessingError as e:
        raise HTTPException(status_code=422, detail=e.detail)
