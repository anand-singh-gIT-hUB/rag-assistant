"""
app/ingestion/pipeline.py
──────────────────────────
Orchestrates: parse → clean → chunk → embed → store.
Returns a list of ChunkMetadata objects (one per stored chunk).
"""
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import Settings
from app.core.logging import get_logger
from app.embeddings.base import EmbedderBase
from app.ingestion.cleaner import clean_pages
from app.ingestion.loaders.base import PageContent
from app.ingestion.parser import parse_file
from app.processing.chunker import RecursiveTextSplitter
from app.processing.metadata import ChunkMetadata
from app.utils.id_utils import new_chunk_id
from app.vectorstore.base import VectorStoreBase

logger = get_logger(__name__)


class IngestionPipeline:
    def __init__(
        self,
        embedder: EmbedderBase,
        vector_store: VectorStoreBase,
        settings: Settings,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.splitter = RecursiveTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    def run(
        self,
        path: Path,
        doc_id: str,
        file_name: str,
        source_type: str,
        uploaded_at: datetime | None = None,
    ) -> list[ChunkMetadata]:
        if uploaded_at is None:
            uploaded_at = datetime.now(timezone.utc)

        logger.info("Ingestion started", doc_id=doc_id, file=file_name)

        # 1. Parse
        pages: list[PageContent] = parse_file(path)
        # 2. Clean
        pages = clean_pages(pages)
        # 3. Chunk
        all_chunks: list[ChunkMetadata] = []
        raw_chunks: list[str] = []

        for page in pages:
            for raw in self.splitter.split(page.text):
                raw_chunks.append(raw)
                all_chunks.append(
                    ChunkMetadata(
                        chunk_id="",          # filled below
                        doc_id=doc_id,
                        file_name=file_name,
                        source_type=source_type,
                        chunk_index=len(all_chunks),
                        total_chunks=0,       # filled below
                        text=raw,
                        page_number=page.page_number,
                        section_title=page.section_title,
                        uploaded_at=uploaded_at,
                    )
                )

        total = len(all_chunks)
        for i, chunk in enumerate(all_chunks):
            chunk.chunk_id = new_chunk_id(doc_id, i)
            chunk.total_chunks = total

        logger.info("Chunking done", doc_id=doc_id, num_chunks=total)

        # 4. Embed
        embeddings = self.embedder.embed_batch([c.text for c in all_chunks])
        logger.info("Embedding done", doc_id=doc_id, num_embeddings=len(embeddings))

        # 5. Store
        self.vector_store.add_chunks(chunks=all_chunks, embeddings=embeddings)
        logger.info("Stored in vector DB", doc_id=doc_id)

        return all_chunks
