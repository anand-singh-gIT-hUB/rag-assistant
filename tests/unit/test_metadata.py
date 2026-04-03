"""
tests/unit/test_metadata.py
"""
from datetime import datetime, timezone
from app.processing.metadata import ChunkMetadata


def test_to_chroma_metadata_basic():
    chunk = ChunkMetadata(
        chunk_id="doc_abc_chunk_00001",
        doc_id="doc_abc",
        file_name="test.pdf",
        source_type="pdf",
        chunk_index=1,
        total_chunks=10,
        text="Sample text",
        page_number=2,
        section_title="Introduction",
        uploaded_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    meta = chunk.to_chroma_metadata()
    assert meta["doc_id"] == "doc_abc"
    assert meta["page_number"] == 2
    assert meta["section_title"] == "Introduction"
    assert "text" not in meta   # text is stored separately


def test_missing_page_number_defaults_to_minus_one():
    chunk = ChunkMetadata(
        chunk_id="x", doc_id="d", file_name="f.txt",
        source_type="txt", chunk_index=0, total_chunks=1, text="hi"
    )
    meta = chunk.to_chroma_metadata()
    assert meta["page_number"] == -1
