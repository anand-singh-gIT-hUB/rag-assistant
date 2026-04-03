"""
app/utils/id_utils.py
"""
import uuid


def new_doc_id() -> str:
    """Generate a new document ID."""
    return f"doc_{uuid.uuid4().hex}"


def new_chunk_id(doc_id: str, index: int) -> str:
    """Generate a deterministic chunk ID."""
    return f"{doc_id}_chunk_{index:05d}"


def new_run_id() -> str:
    """Generate a new evaluation run ID."""
    return f"run_{uuid.uuid4().hex[:8]}"
