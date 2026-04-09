"""
app/api/routes/evaluation.py
─────────────────────────────
POST /evaluate/run   — run Ragas benchmark.
GET  /evaluate/results — fetch stored results.
"""
from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_evaluation_service
from app.core.exceptions import EvaluationError
from app.schemas.evaluation import (
    EvaluationRunRequest,
    EvaluationRunResponse,
    EvaluationResultsResponse,
)
from app.services.evaluation_service import EvaluationService

router = APIRouter()


@router.post(
    "/run",
    response_model=EvaluationRunResponse,
    summary="Run Ragas evaluation benchmark",
)
def run_evaluation(
    request: EvaluationRunRequest,
    service: EvaluationService = Depends(get_evaluation_service),
) -> EvaluationRunResponse:
    try:
        return service.run_benchmark(mode=request.mode)
    except EvaluationError as e:
        raise HTTPException(status_code=500, detail=e.detail)


@router.get(
    "/results",
    response_model=EvaluationResultsResponse,
    summary="Retrieve stored evaluation results",
)
def get_results(
    service: EvaluationService = Depends(get_evaluation_service),
) -> EvaluationResultsResponse:
    return service.get_results()
