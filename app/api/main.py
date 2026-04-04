"""
app/api/main.py
───────────────
FastAPI application factory.  Registers all routers, CORS middleware,
global exception handlers, and lifespan events.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.exceptions import RAGBaseException
from app.core.logging import get_logger, setup_logging
from app.api.routes import health, documents, query, evaluation
from app.api.dependencies import _build_query_service

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown events."""
    setup_logging()
    settings = get_settings()
    # Ensure storage directories exist
    _ = settings.files_path
    _ = settings.logs_path
    _ = settings.chroma_persist_path
    logger.info("RAG Assistant API starting", provider=settings.llm_provider)
    # ── Force-load ALL heavy components at startup so first user query is instant ─
    import asyncio
    loop = asyncio.get_event_loop()
    svc = _build_query_service()  # builds pipeline singleton (reranker only if enabled)

    # 1. Warm embedder: loads SentenceTransformer weights into RAM
    try:
        embedder = svc._pipeline._retriever.embedder
        await loop.run_in_executor(None, embedder.embed_query, "warmup")
        logger.info("Embedder warmup complete")
    except Exception as e:
        logger.warning("Embedder warmup failed (non-fatal)", error=str(e))

    # 2. Warm LLM: loads Ollama model into memory
    try:
        await loop.run_in_executor(
            None, svc._llm.complete,
            "You are a helpful assistant.", "Reply with: ready"
        )
        logger.info("LLM warmup complete")
    except Exception as e:
        logger.warning("LLM warmup failed (non-fatal)", error=str(e))

    yield
    logger.info("RAG Assistant API shutting down")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="RAG Assistant API",
        description=(
            "Production-grade Retrieval-Augmented Generation backend.  "
            "Upload documents, ask questions, get grounded cited answers."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Global exception handler ───────────────────────────────────────────────
    @app.exception_handler(RAGBaseException)
    async def rag_exception_handler(request: Request, exc: RAGBaseException):
        logger.error("RAG error", detail=exc.detail, path=str(request.url))
        return JSONResponse(status_code=500, content={"detail": exc.detail})

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(health.router)
    app.include_router(documents.router, prefix="/documents", tags=["documents"])
    app.include_router(query.router, tags=["query"])
    app.include_router(evaluation.router, prefix="/evaluate", tags=["evaluation"])

    return app


app = create_app()
