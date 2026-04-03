# RAG Knowledge Assistant

A **production-grade, modular Retrieval-Augmented Generation (RAG) system** built in Python.

Upload documents → they are parsed, chunked, embedded and stored in ChromaDB → a FastAPI backend handles queries → an LLM produces grounded, cited answers → a Streamlit UI exposes everything.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY (or switch to EMBEDDING_PROVIDER=huggingface + LLM_PROVIDER=ollama)
```

### 3. Start the API

```bash
uvicorn app.api.main:app --reload --port 8000
```

### 4. Start the UI (new terminal)

```bash
streamlit run streamlit_app/Home.py
```

Open **http://localhost:8501** in your browser.

---

## Tech Stack

| Layer | Default | Swap-In |
|---|---|---|
| Backend API | FastAPI + Uvicorn | — |
| Embeddings | OpenAI `text-embedding-3-small` | HuggingFace `all-MiniLM-L6-v2` |
| Vector DB | ChromaDB | Qdrant (stub) |
| LLM | OpenAI `gpt-4.1-mini` | Ollama |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | — |
| Frontend | Streamlit multi-page | — |
| Evaluation | Ragas | — |
| Containers | Docker + docker-compose | — |
| Tests | pytest | — |

---

## Directory Structure

```
rag-assistant/
├── app/
│   ├── api/            # FastAPI app, routes, dependencies
│   ├── core/           # Config, logging, constants, exceptions
│   ├── ingestion/      # Loaders, parser, cleaner, pipeline
│   ├── processing/     # Chunker, metadata, tokenizer
│   ├── embeddings/     # OpenAI + HF embedders, factory
│   ├── vectorstore/    # ChromaDB + Qdrant, factory
│   ├── retrieval/      # Dense retriever, reranker, pipeline
│   ├── llm/            # OpenAI + Ollama LLMs, prompts, factory
│   ├── services/       # Document, Query, Citation, Evaluation services
│   ├── schemas/        # Pydantic request/response models
│   ├── storage/        # files/, vectordb/, logs/
│   └── utils/          # file_utils, text_utils, id_utils
├── streamlit_app/
│   ├── Home.py
│   ├── pages/          # Chat, Documents, Settings, Evaluation
│   ├── components/     # chat_box, uploader, sidebar, source_viewer
│   └── utils/          # api_client, session
├── evaluation/         # benchmark_questions.json, ground_truth.json
├── tests/              # unit/ and integration/
├── scripts/            # ingest_sample.py, run_evaluation.py
├── docker/             # Dockerfile.api, Dockerfile.streamlit
├── docker-compose.yml
├── Makefile
└── requirements.txt
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Service health |
| POST | `/documents/upload` | Upload + index a document |
| GET | `/documents` | List indexed documents |
| DELETE | `/documents/{doc_id}` | Delete document + vectors |
| POST | `/documents/{doc_id}/reindex` | Re-parse and re-embed |
| POST | `/query` | Ask a question, get grounded answer |
| POST | `/retrieve` | Debug: return raw chunks only |
| POST | `/evaluate/run` | Run Ragas benchmark |
| GET | `/evaluate/results` | Get stored evaluation results |

Interactive docs: **http://localhost:8000/docs**

---

## Configuration

All settings are read from environment variables (or `.env`):

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai` or `ollama` |
| `OPENAI_API_KEY` | — | Required for OpenAI provider |
| `EMBEDDING_PROVIDER` | `openai` | `openai` or `huggingface` |
| `VECTOR_STORE_PROVIDER` | `chroma` | `chroma` or `qdrant` |
| `RERANKER_ENABLED` | `true` | Toggle cross-encoder reranking |
| `CHUNK_SIZE` | `512` | Token budget per chunk |
| `CHUNK_OVERLAP` | `64` | Token overlap between chunks |
| `RETRIEVAL_TOP_K` | `20` | Candidates retrieved before reranking |
| `RERANK_TOP_N` | `5` | Final chunks passed to LLM |

---

## Running Tests

```bash
# Unit tests (no external services needed)
pytest tests/unit/ -v

# All tests
pytest tests/ -v
```

---

## Docker

```bash
docker-compose up --build
# API:  http://localhost:8000/docs
# UI:   http://localhost:8501
```

---

## Switching to Open-Source Stack (No API Key)

```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3
EMBEDDING_PROVIDER=huggingface
HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

Then start Ollama locally:
```bash
ollama serve
ollama pull llama3
```

---

## Key Design Decisions

1. **Provider abstraction via factory pattern** — swap providers by changing one env var.
2. **Chunk metadata is first-class** — every chunk carries `doc_id`, `file_name`, `page_number`, `section_title`, enabling precise citations.
3. **Retrieval is a pipeline** — embed → retrieve → filter → rerank → top-N.
4. **LLM is strictly grounded** — system prompt forbids fabrication; model states ignorance when context is insufficient.
5. **JSONL query log** — all queries appended to `app/storage/logs/queries.jsonl` for offline analysis.
6. **Streamlit talks only to FastAPI** — all business logic in the backend.
