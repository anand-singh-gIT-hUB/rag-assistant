"""
app/services/query_service.py
──────────────────────────────
Business logic for answering questions and raw retrieval.
Logs every query to a JSONL file.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import Settings
from app.core.exceptions import QueryValidationError
from app.core.constants import MAX_QUERY_LENGTH, QUERIES_LOG_FILE
from app.core.logging import get_logger
from app.llm.base import LLMBase
from app.llm.prompts import SYSTEM_PROMPT, build_user_prompt
from app.retrieval.pipeline import RetrievalPipeline
from app.schemas.query import (
    ChunkResult,
    QueryRequest,
    QueryResponse,
    RetrieveRequest,
    RetrieveResponse,
)
from app.services.citation_service import build_citations

logger = get_logger(__name__)


class QueryService:
    def __init__(
        self,
        retrieval_pipeline: RetrievalPipeline,
        llm: LLMBase,
        settings: Settings,
    ) -> None:
        self._pipeline = retrieval_pipeline
        self._llm = llm
        self._settings = settings
        self._log_path: Path = settings.logs_path / QUERIES_LOG_FILE

    def _log_query(self, entry: dict) -> None:
        with self._log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def answer(self, request: QueryRequest) -> QueryResponse:
        if len(request.question) > MAX_QUERY_LENGTH:
            raise QueryValidationError(
                "Query too long",
                detail=f"Max {MAX_QUERY_LENGTH} characters.",
            )

        chunks, reranked = self._pipeline.run(
            query=request.question,
            top_k=request.top_k,
            doc_ids=request.doc_ids,
            rerank=request.rerank,
        )

        user_prompt = build_user_prompt(request.question, chunks)
        answer = self._llm.complete(SYSTEM_PROMPT, user_prompt)
        citations = build_citations(chunks)

        self._log_query(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "question": request.question,
                "model": self._llm.model_name,
                "num_chunks": len(chunks),
                "reranked": reranked,
                "answer_len": len(answer),
            }
        )

        return QueryResponse(
            question=request.question,
            answer=answer,
            citations=citations,
            model=self._llm.model_name,
            retrieval_top_k=len(chunks),
            reranked=reranked,
        )

    def answer_stream(self, request: QueryRequest):
        import time
        if len(request.question) > MAX_QUERY_LENGTH:
            yield json.dumps({"type": "error", "detail": f"Max {MAX_QUERY_LENGTH} characters."}) + "\n"
            return

        t0 = time.perf_counter()
        chunks, reranked = self._pipeline.run(
            query=request.question,
            top_k=request.top_k,
            doc_ids=request.doc_ids,
            rerank=request.rerank,
        )
        retrieval_time = time.perf_counter() - t0
        logger.info("Retrieval complete", retrieve_s=round(retrieval_time, 2), n_chunks=len(chunks))

        citations = build_citations(chunks)
        yield json.dumps({
            "type": "metadata",
            "citations": [c.model_dump() if hasattr(c, 'model_dump') else c.dict() for c in citations] if citations else [],
            "model": self._llm.model_name,
            "retrieval_top_k": len(chunks),
            "reranked": reranked,
            "retrieval_time": round(retrieval_time, 2)
        }) + "\n"

        user_prompt = build_user_prompt(request.question, chunks)

        t1 = time.perf_counter()
        answer = ""
        first_token_time = None
        try:
            for token in self._llm.stream_complete(SYSTEM_PROMPT, user_prompt):
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                    logger.info("First token received", ttft_s=round(first_token_time - t1, 2))
                answer += token
                yield json.dumps({"type": "token", "content": token}) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "detail": str(e)}) + "\n"
            return

        gen_time = time.perf_counter() - t1
        total_time = time.perf_counter() - t0
        logger.info(
            "Query complete",
            retrieve_s=round(retrieval_time, 2),
            first_token_s=round(first_token_time - t1, 2) if first_token_time else None,
            gen_s=round(gen_time, 2),
            total_s=round(total_time, 2),
        )

        self._log_query({
            "ts": datetime.now(timezone.utc).isoformat(),
            "question": request.question,
            "model": self._llm.model_name,
            "num_chunks": len(chunks),
            "reranked": reranked,
            "answer_len": len(answer),
            "retrieval_s": round(retrieval_time, 2),
            "gen_s": round(gen_time, 2),
            "total_s": round(total_time, 2),
        })

        yield json.dumps({"type": "done", "gen_time": round(gen_time, 2), "total_time": round(total_time, 2)}) + "\n"

    def retrieve_only(self, request: RetrieveRequest) -> RetrieveResponse:
        chunks, _ = self._pipeline.run(
            query=request.question,
            top_k=request.top_k,
            doc_ids=request.doc_ids,
            rerank=False,
        )
        results = [
            ChunkResult(
                chunk_id=c.get("chunk_id", ""),
                doc_id=c.get("metadata", {}).get("doc_id", ""),
                file_name=c.get("metadata", {}).get("file_name", ""),
                page_number=c.get("metadata", {}).get("page_number"),
                section_title=c.get("metadata", {}).get("section_title"),
                text=c.get("text", ""),
                score=c.get("score", 0.0),
            )
            for c in chunks
        ]
        return RetrieveResponse(question=request.question, chunks=results)
