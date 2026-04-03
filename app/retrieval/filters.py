"""
app/retrieval/filters.py
─────────────────────────
Build ChromaDB `where` filter dicts from user-supplied doc_ids.
"""


def build_doc_filter(doc_ids: list[str] | None) -> dict | None:
    """Return a ChromaDB where-filter or None if no filter needed."""
    if not doc_ids:
        return None
    if len(doc_ids) == 1:
        return {"doc_id": {"$eq": doc_ids[0]}}
    return {"doc_id": {"$in": doc_ids}}
