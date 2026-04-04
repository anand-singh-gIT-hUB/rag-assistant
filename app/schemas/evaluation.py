"""
app/schemas/evaluation.py
"""
from datetime import datetime
from typing import Literal
from pydantic import BaseModel


class EvaluationRunRequest(BaseModel):
    mode: Literal["fast", "full"] = "fast"


class EvaluationRunResponse(BaseModel):
    run_id: str
    started_at: datetime
    finished_at: datetime
    num_questions: int
    metrics: dict[str, float]
    status: str
    mode: str


class EvaluationResultsResponse(BaseModel):
    results: list[EvaluationRunResponse]
    total: int
