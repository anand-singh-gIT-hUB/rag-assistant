"""
app/services/citation_service.py
──────────────────────────────────
Builds Citation objects from retrieved chunk dicts.
"""
from app.schemas.query import Citation
from app.utils.text_utils import truncate


def build_citations(chunks: list[dict]) -> list[Citation]:
    citations: list[Citation] = []
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        page = meta.get("page_number", -1)
        citations.append(
            Citation(
                chunk_id=chunk.get("chunk_id", ""),
                doc_id=meta.get("doc_id", ""),
                file_name=meta.get("file_name", "unknown"),
                page_number=page if page and page != -1 else None,
                section_title=meta.get("section_title") or None,
                excerpt=truncate(chunk.get("text", ""), max_chars=250),
            )
        )
    return citations
