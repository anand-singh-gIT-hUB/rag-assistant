"""
app/services/document_service.py
──────────────────────────────────
Business logic for document ingestion, listing, deletion, and reindexing.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import UploadFile

from app.core.config import Settings
from app.core.exceptions import DocumentNotFoundError, StorageError
from app.core.logging import get_logger
from app.embeddings.base import EmbedderBase
from app.ingestion.pipeline import IngestionPipeline
from app.schemas.document import DocumentListResponse, DocumentMeta, DocumentResponse
from app.utils.file_utils import save_upload, validate_extension, safe_delete
from app.utils.id_utils import new_doc_id
from app.vectorstore.base import VectorStoreBase

logger = get_logger(__name__)

# Simple JSON registry for document metadata
_REGISTRY_FILE = "app/storage/files/registry.json"


class DocumentService:
    def __init__(
        self,
        embedder: EmbedderBase,
        vector_store: VectorStoreBase,
        settings: Settings,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._settings = settings
        self._registry_path = settings.files_path / "registry.json"
        self._pipeline = IngestionPipeline(
            embedder=embedder, vector_store=vector_store, settings=settings
        )

    # ── Registry helpers ──────────────────────────────────────────────────────

    def _load_registry(self) -> dict[str, dict]:
        if self._registry_path.exists():
            return json.loads(self._registry_path.read_text())
        return {}

    def _save_registry(self, registry: dict[str, dict]) -> None:
        self._registry_path.write_text(json.dumps(registry, indent=2, default=str))

    # ── Public API ────────────────────────────────────────────────────────────

    async def ingest(self, file: UploadFile) -> DocumentResponse:
        content = await file.read()
        ext = validate_extension(file.filename or "unknown")
        source_type = ext.lstrip(".")
        doc_id = new_doc_id()
        uploaded_at = datetime.now(timezone.utc)
        dest = self._settings.files_path / f"{doc_id}{ext}"

        save_upload(content, dest)
        logger.info("File saved", doc_id=doc_id, path=str(dest))

        chunks = self._pipeline.run(
            path=dest,
            doc_id=doc_id,
            file_name=file.filename or dest.name,
            source_type=source_type,
            uploaded_at=uploaded_at,
        )

        record = {
            "doc_id": doc_id,
            "file_name": file.filename,
            "file_type": source_type,
            "file_size_bytes": len(content),
            "num_chunks": len(chunks),
            "uploaded_at": uploaded_at.isoformat(),
            "file_path": str(dest),
        }
        registry = self._load_registry()
        registry[doc_id] = record
        self._save_registry(registry)

        return DocumentResponse(**record, status="indexed")

    async def list_documents(self) -> DocumentListResponse:
        registry = self._load_registry()
        docs = [DocumentMeta(**{k: v for k, v in r.items() if k != "file_path"})
                for r in registry.values()]
        return DocumentListResponse(documents=docs, total=len(docs))

    async def delete(self, doc_id: str) -> None:
        registry = self._load_registry()
        if doc_id not in registry:
            raise DocumentNotFoundError(f"Document {doc_id} not found")
        record = registry.pop(doc_id)
        self._vector_store.delete_by_doc_id(doc_id)
        safe_delete(Path(record.get("file_path", "")))
        self._save_registry(registry)
        logger.info("Document deleted", doc_id=doc_id)

    async def reindex(self, doc_id: str) -> DocumentResponse:
        registry = self._load_registry()
        if doc_id not in registry:
            raise DocumentNotFoundError(f"Document {doc_id} not found")
        record = registry[doc_id]

        self._vector_store.delete_by_doc_id(doc_id)
        chunks = self._pipeline.run(
            path=Path(record["file_path"]),
            doc_id=doc_id,
            file_name=record["file_name"],
            source_type=record["file_type"],
        )
        record["num_chunks"] = len(chunks)
        self._save_registry(registry)
        return DocumentResponse(**{k: v for k, v in record.items() if k != "file_path"}, status="reindexed")
