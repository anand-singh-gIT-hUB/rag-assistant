"""
streamlit_app/utils/api_client.py
───────────────────────────────────
Thin httpx-based client that wraps all FastAPI endpoints.
"""
import httpx
from typing import Any

BASE_URL = "http://127.0.0.1:8000"
TIMEOUT = 900.0


def _get(path: str, **params) -> dict:
    with httpx.Client(timeout=TIMEOUT) as client:
        r = client.get(f"{BASE_URL}{path}", params=params)
        r.raise_for_status()
        return r.json()


def _post(path: str, json: Any = None, files=None, data=None) -> dict:
    with httpx.Client(timeout=TIMEOUT) as client:
        r = client.post(f"{BASE_URL}{path}", json=json, files=files, data=data)
        r.raise_for_status()
        return r.json()


def _delete(path: str) -> None:
    with httpx.Client(timeout=TIMEOUT) as client:
        r = client.delete(f"{BASE_URL}{path}")
        r.raise_for_status()


# ── Documents ──────────────────────────────────────────────────────────────────

def upload_document(file_bytes: bytes, filename: str) -> dict:
    return _post("/documents/upload", files={"file": (filename, file_bytes)})


def list_documents() -> dict:
    return _get("/documents")


def delete_document(doc_id: str) -> None:
    _delete(f"/documents/{doc_id}")


def reindex_document(doc_id: str) -> dict:
    return _post(f"/documents/{doc_id}/reindex")


# ── Query ──────────────────────────────────────────────────────────────────────

def query(question: str, top_k: int | None = None, doc_ids: list[str] | None = None, rerank: bool | None = None) -> dict:
    payload: dict = {"question": question}
    if top_k is not None:
        payload["top_k"] = top_k
    if doc_ids:
        payload["doc_ids"] = doc_ids
    if rerank is not None:
        payload["rerank"] = rerank
    return _post("/query", json=payload)


def retrieve(question: str, top_k: int = 10) -> dict:
    return _post("/retrieve", json={"question": question, "top_k": top_k})


# ── Evaluation ────────────────────────────────────────────────────────────────

def run_evaluation() -> dict:
    return _post("/evaluate/run")


def get_evaluation_results() -> dict:
    return _get("/evaluate/results")


# ── Health ────────────────────────────────────────────────────────────────────

def health_check() -> dict:
    return _get("/health")
