"""
app/processing/metadata.py
───────────────────────────
ChunkMetadata dataclass — first-class metadata for every chunk.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class ChunkMetadata:
    chunk_id: str
    doc_id: str
    file_name: str
    source_type: str                      # pdf | docx | txt | markdown
    chunk_index: int
    total_chunks: int
    text: str
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    uploaded_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def to_chroma_metadata(self) -> dict:
        """Serialise to a flat dict compatible with ChromaDB metadata."""
        return {
            "doc_id": self.doc_id,
            "file_name": self.file_name,
            "source_type": self.source_type,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "page_number": self.page_number or -1,
            "section_title": self.section_title or "",
            "uploaded_at": self.uploaded_at.isoformat(),
        }
