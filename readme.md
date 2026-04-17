# RAG Knowledge Assistant

A **production-grade, modular Retrieval-Augmented Generation (RAG) system** built in Python.

Upload documents → they are parsed, chunked, embedded and stored in ChromaDB → a FastAPI backend handles queries → an LLM produces grounded, cited answers → a Streamlit UI exposes everything.

---

## Quick Start

### 1. Install dependencies
 
 ```bash
 # For full local development (Backend + UI + Evaluation)
 pip install -r requirements-dev.txt
 ```

### 2. Configure environment
 
 ```bash
 cp .env.example .env
 # Set APP_ENV=dev for local features.
 # Set your OPENAI_API_KEY if using OpenAI providers.
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

## 🚀 Deployment & Environments

The project supports two distinct modes to balance performance and features:

| Feature | **Development** (`dev`) | **Production** (`prod`) |
|---|---|---|
| **Target Context** | Local Laptop / Desktop | Railway / Cloud (CPU) |
| **Evaluation (Ragas)** | ✅ Enabled | ❌ Disabled by default |
| **Reranking** | ✅ Enabled | ❌ Disabled by default |
| **Dependencies** | `requirements-dev.txt` | `requirements.txt` |
| **PyTorch** | Standard | CPU-Optimized |

### Railway Deployment (Backend)
1. Connect your repo to Railway.
2. Set environment variables: `APP_ENV=prod`, `OPENAI_API_KEY=...`.
3. The `docker/Dockerfile.api` will handle the lightweight CPU-only build automatically using the production `requirements.txt`.

### Local Evaluation
To run Ragas benchmarks locally, ensure you have installed the evaluation extensions:
```bash
pip install -r requirements-eval.txt
```

---

## 📂 Directory Structure

```
rag-assistant/
├── app/
├── streamlit_app/
├── evaluation/
├── tests/
├── scripts/
├── docker/
├── docker-compose.yml
├── Makefile
├── requirements.txt       # Production core
├── requirements-eval.txt  # Evaluation extensions
└── requirements-dev.txt   # Full dev setup
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
| POST | `/evaluate/run` | Run Ragas benchmark (Dev only) |
| GET | `/evaluate/results` | Get stored evaluation results |

Interactive docs: **http://localhost:8000/docs**

---

## Configuration

All settings are read from environment variables (or `.env`):

| Variable | Default | Description |
|---|---|---|
| `APP_ENV` | `dev` | `dev` or `prod` |
| `ENABLE_EVALUATION` | `true` | Set `false` to skip eval deps |
| `LLM_PROVIDER` | `openai` | `openai` or `ollama` |
| `RERANKER_ENABLED` | `true` | Toggle cross-encoder reranking |

---

## Key Design Decisions

1. **Provider abstraction via factory pattern** — swap providers by changing one env var.
2. **Chunk metadata is first-class** — every chunk carries `doc_id`, `file_name`, etc.
3. **Retrieval is a pipeline** — embed → retrieve → filter → rerank → top-N.
4. **Environment-aware scaling** — production builds skip heavy ML deps to ensure cloud stability.
