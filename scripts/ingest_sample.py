"""
scripts/ingest_sample.py
─────────────────────────
Seed script: creates a sample Markdown document and ingests it via the API.
Run: python scripts/ingest_sample.py
"""
import httpx
from pathlib import Path

SAMPLE_MD = """\
# RAG Assistant — System Overview

## What is RAG?
Retrieval-Augmented Generation (RAG) combines a retrieval system and a
generative language model. Instead of relying solely on model parameters,
RAG fetches relevant passages from a document corpus and uses them as
grounded context for the LLM to produce accurate, cited answers.

## Architecture
The system is composed of the following layers:
- **Ingestion Pipeline** — parses, cleans, chunks, embeds, and stores documents.
- **Retrieval Pipeline** — embeds the query, performs dense retrieval, optionally reranks.
- **LLM Layer** — generates an answer from the grounded context.
- **FastAPI Backend** — exposes REST endpoints for all operations.
- **Streamlit Frontend** — thin UI talking exclusively to the API.

## Supported File Formats
PDF, DOCX, TXT, and Markdown files are supported out of the box.

## Embedding Models
- **OpenAI** `text-embedding-3-small` (default, 1536 dims)
- **HuggingFace** `sentence-transformers/all-MiniLM-L6-v2` (open-source, 384 dims)

## Reranking
A cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) reranks the
top-K retrieval candidates before passing context to the LLM.

## Evaluation
Ragas metrics — faithfulness, answer relevancy, context precision, and
context recall — measure pipeline quality against a benchmark question set.
"""

def main():
    sample_path = Path("evaluation/sample_doc.md")
    sample_path.write_text(SAMPLE_MD, encoding="utf-8")
    print(f"Created {sample_path}")

    with httpx.Client(timeout=60) as client:
        with open(sample_path, "rb") as f:
            r = client.post(
                "http://localhost:8000/documents/upload",
                files={"file": ("sample_doc.md", f, "text/markdown")},
            )
        r.raise_for_status()
        data = r.json()
        print(f"Ingested: {data['file_name']} — {data['num_chunks']} chunks (doc_id: {data['doc_id']})")


if __name__ == "__main__":
    main()
