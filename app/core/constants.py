"""
app/core/constants.py
─────────────────────
Application-wide immutable constants.
"""
from pathlib import Path

# ── Supported document MIME types ────────────────────────────────────────────
SUPPORTED_MIME_TYPES: dict[str, str] = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "text/plain": "txt",
    "text/markdown": "markdown",
    "text/x-markdown": "markdown",
}

SUPPORTED_EXTENSIONS: set[str] = {".pdf", ".docx", ".txt", ".md", ".markdown"}

# ── Query ────────────────────────────────────────────────────────────────────
MAX_QUERY_LENGTH: int = 2_000
MAX_CONTEXT_TOKENS: int = 6_000

# ── Chunking defaults ────────────────────────────────────────────────────────
DEFAULT_CHUNK_SIZE: int = 512
DEFAULT_CHUNK_OVERLAP: int = 64

# ── Storage sub-dirs ─────────────────────────────────────────────────────────
QUERIES_LOG_FILE: str = "queries.jsonl"
EVAL_RESULTS_FILE: str = "eval_results.jsonl"

# ── API tags ─────────────────────────────────────────────────────────────────
TAG_HEALTH = "health"
TAG_DOCUMENTS = "documents"
TAG_QUERY = "query"
TAG_EVALUATION = "evaluation"
