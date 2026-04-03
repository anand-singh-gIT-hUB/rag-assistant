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

logger = get_logger(__name__)

_BENCHMARK_FILE = Path("evaluation/benchmark_questions.json")
_GROUND_TRUTH_FILE = Path("evaluation/ground_truth.json")


class EvaluationService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._results_path = settings.logs_path / EVAL_RESULTS_FILE

    async def run_benchmark(self) -> EvaluationRunResponse:
        started_at = datetime.now(timezone.utc)
        run_id = f"run_{uuid.uuid4().hex[:8]}"

        if not _BENCHMARK_FILE.exists():
            raise EvaluationError(
                "Benchmark file not found",
                detail=str(_BENCHMARK_FILE),
            )

        questions_data = json.loads(_BENCHMARK_FILE.read_text())
        ground_truth_data: dict = {}
        if _GROUND_TRUTH_FILE.exists():
            ground_truth_data = json.loads(_GROUND_TRUTH_FILE.read_text())

        try:
            metrics = await self._run_ragas(questions_data, ground_truth_data)
        except Exception as e:
            raise EvaluationError(f"Ragas evaluation failed: {e}") from e

        finished_at = datetime.now(timezone.utc)
        result = EvaluationRunResponse(
            run_id=run_id,
            started_at=started_at,
            finished_at=finished_at,
            num_questions=len(questions_data),
            metrics=metrics,
            status="completed",
        )

        # Append to JSONL log
        with self._results_path.open("a", encoding="utf-8") as f:
            f.write(result.model_dump_json() + "\n")

        logger.info("Evaluation completed", run_id=run_id, metrics=metrics)
        return result

    async def _run_ragas(self, questions: list, ground_truth: dict) -> dict[str, float]:
        """Run Ragas metrics. Returns metric scores."""
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        )

        rows = []
        for item in questions:
            q = item.get("question", "")
            rows.append(
                {
                    "question": q,
                    "answer": item.get("answer", ""),
                    "contexts": item.get("contexts", []),
                    "ground_truth": ground_truth.get(q, ""),
                }
            )

        dataset = Dataset.from_list(rows)
        result = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_recall,
                context_precision,
            ],
        )
        return {k: float(v) for k, v in result.items() if isinstance(v, (int, float))}

    async def get_results(self) -> EvaluationResultsResponse:
        if not self._results_path.exists():
            return EvaluationResultsResponse(results=[], total=0)

        results = []
        for line in self._results_path.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    results.append(EvaluationRunResponse.model_validate_json(line))
                except Exception:
                    pass

        return EvaluationResultsResponse(results=results, total=len(results))
