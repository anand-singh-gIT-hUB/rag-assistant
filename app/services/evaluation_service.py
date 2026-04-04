"""
app/services/evaluation_service.py
────────────────────────────────────
Runs Ragas benchmark evaluation and stores results.
"""
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import Settings
from app.core.constants import EVAL_RESULTS_FILE
from app.core.exceptions import EvaluationError
from app.core.logging import get_logger
from app.schemas.evaluation import EvaluationResultsResponse, EvaluationRunResponse
from app.schemas.query import QueryRequest
from app.services.query_service import QueryService

logger = get_logger(__name__)

_BENCHMARK_FILE = Path("evaluation/benchmark_questions.json")
_GROUND_TRUTH_FILE = Path("evaluation/ground_truth.json")


class EvaluationService:
    def __init__(self, settings: Settings, query_service: QueryService) -> None:
        self._settings = settings
        self._query_service = query_service
        self._results_path = settings.logs_path / EVAL_RESULTS_FILE

    def run_benchmark(self, mode: str = "fast") -> EvaluationRunResponse:
        started_at = datetime.now(timezone.utc)
        run_id = f"run_{uuid.uuid4().hex[:8]}"

        if not _BENCHMARK_FILE.exists():
            raise EvaluationError("Benchmark file not found", detail=str(_BENCHMARK_FILE))

        try:
            questions_data = json.loads(_BENCHMARK_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            raise EvaluationError("Failed to read benchmark file", detail=str(e)) from e

        ground_truth_data: dict = {}
        if _GROUND_TRUTH_FILE.exists():
            try:
                ground_truth_data = json.loads(_GROUND_TRUTH_FILE.read_text(encoding="utf-8"))
            except Exception as e:
                raise EvaluationError("Failed to read ground truth file", detail=str(e)) from e

        if mode == "fast" or mode == "full":
            pass # Keep all questions for both modes, fast mode only reduces metrics
        else:
            raise EvaluationError("Invalid evaluation mode", detail=f"Unsupported mode: {mode}")

        try:
            metrics, evaluated_count = self._run_ragas(questions_data, ground_truth_data, mode=mode)
        except Exception as e:
            logger.exception("Evaluation failed")
            raise EvaluationError("Ragas evaluation failed", detail=str(e)) from e

        finished_at = datetime.now(timezone.utc)

        result = EvaluationRunResponse(
            run_id=run_id,
            started_at=started_at,
            finished_at=finished_at,
            num_questions=evaluated_count,
            metrics=metrics,
            status="completed",
            mode=mode,
        )

        with self._results_path.open("a", encoding="utf-8") as f:
            f.write(result.model_dump_json() + "\n")

        logger.info("Evaluation completed", run_id=run_id, mode=mode, metrics=metrics)
        return result

    def _run_ragas(
        self,
        questions: list,
        ground_truth: dict,
        mode: str = "fast",
    ) -> tuple[dict[str, float], int]:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        )
        from langchain_community.chat_models import ChatOllama
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper

        # Local evaluator LLM using Ollama (use qwen2.5:1.5b for speed)
        evaluator_llm = LangchainLLMWrapper(
            ChatOllama(
                model="qwen2.5:1.5b",
                temperature=0,
            )
        )

        # Local embeddings
        evaluator_embeddings = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(
                model_name=self._settings.hf_embedding_model,
            )
        )

        rows = []
        skipped = 0

        for item in questions:
            q = item.get("question", "").strip()
            if not q:
                skipped += 1
                continue

            request = QueryRequest(
                question=q,
                top_k=self._settings.retrieval_top_k,
                rerank=self._settings.reranker_enabled,
                stream=False,
            )

            response = self._query_service.answer(request)

            contexts = [c.excerpt for c in response.citations] if response.citations else []

            if not response.answer or not contexts:
                logger.warning(
                    "Skipping evaluation row due to empty answer or contexts",
                    question=q,
                    has_answer=bool(response.answer),
                    n_contexts=len(contexts),
                )
                skipped += 1
                continue

            rows.append(
                {
                    "question": q,
                    "answer": response.answer,
                    "contexts": contexts,
                    "ground_truth": ground_truth.get(q, ""),
                }
            )

        if not rows:
            raise EvaluationError(
                "No evaluation rows were generated",
                detail="All benchmark questions returned empty answers or empty contexts."
            )

        logger.info(
            "Prepared evaluation dataset",
            n_rows=len(rows),
            skipped=skipped,
            sample_question=rows[0]["question"],
            sample_has_answer=bool(rows[0]["answer"]),
            sample_n_contexts=len(rows[0]["contexts"]),
        )

        dataset = Dataset.from_list(rows)

        if mode == "fast":
            metrics_to_run = [
                faithfulness,
                answer_relevancy,
            ]
        elif mode == "full":
            metrics_to_run = [
                faithfulness,
                answer_relevancy,
                context_recall,
                context_precision,
            ]
        else:
            raise EvaluationError("Invalid evaluation mode", detail=f"Unsupported mode: {mode}")

        logger.info(
            "Running evaluation",
            mode=mode,
            n_questions=len(questions),
            metrics=[m.name for m in metrics_to_run],
        )

        result = evaluate(
            dataset,
            metrics=metrics_to_run,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        )

        metrics = {k: float(v) for k, v in result.items() if isinstance(v, (int, float))}
        return metrics, len(rows)

    def get_results(self) -> EvaluationResultsResponse:
        if not self._results_path.exists():
            return EvaluationResultsResponse(results=[], total=0)

        results = []
        for line in self._results_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    results.append(EvaluationRunResponse.model_validate_json(line))
                except Exception:
                    pass

        return EvaluationResultsResponse(results=results, total=len(results))
