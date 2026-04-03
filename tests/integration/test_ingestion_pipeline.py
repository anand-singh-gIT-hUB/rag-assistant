"""
tests/integration/test_ingestion_pipeline.py
─────────────────────────────────────────────
Integration test: runs the full ingestion pipeline on a tiny TXT file.
Requires no external services (uses mocked embedder and in-memory store).
"""
import pytest
from pathlib import Path
from unittest.mock import MagicMock


@pytest.fixture
def tmp_txt_file(tmp_path: Path) -> Path:
    f = tmp_path / "sample.txt"
    f.write_text("This is a test document.\n\nIt has two paragraphs.")
    return f


def test_ingestion_pipeline_end_to_end(tmp_txt_file):
    from app.ingestion.pipeline import IngestionPipeline

    settings = MagicMock()
    settings.chunk_size = 50
    settings.chunk_overlap = 10

    embedder = MagicMock()
    embedder.embed_batch.return_value = [[0.1] * 3]

    store = MagicMock()

    pipeline = IngestionPipeline(
        embedder=embedder, vector_store=store, settings=settings
    )
    chunks = pipeline.run(
        path=tmp_txt_file,
        doc_id="test_doc",
        file_name="sample.txt",
        source_type="txt",
    )

    assert len(chunks) >= 1
    store.add_chunks.assert_called_once()
    embedder.embed_batch.assert_called_once()
