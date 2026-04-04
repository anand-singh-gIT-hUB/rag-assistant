"""
app/api/routes/query.py
────────────────────────
POST /query  — answer a question with grounded citations.
POST /retrieve — debug: return raw chunks only.
"""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.api.dependencies import get_query_service
from app.core.exceptions import QueryValidationError, LLMError
from app.schemas.query import QueryRequest, QueryResponse, RetrieveRequest, RetrieveResponse
from app.services.query_service import QueryService

router = APIRouter()


@router.post("/query", summary="Ask a question")
def query_endpoint(
    request: QueryRequest,
    service: QueryService = Depends(get_query_service),
):
    try:
        if request.stream:
            return StreamingResponse(
                service.answer_stream(request),
                media_type="application/x-ndjson",
            )
        return service.answer(request)
    except QueryValidationError as e:
        raise HTTPException(status_code=400, detail=e.detail)
    except LLMError as e:
        raise HTTPException(status_code=502, detail=e.detail)


@router.post(
    "/retrieve",
    response_model=RetrieveResponse,
    summary="Debug: retrieve raw chunks only",
)
def retrieve(
    request: RetrieveRequest,
    service: QueryService = Depends(get_query_service),
) -> RetrieveResponse:
    return service.retrieve_only(request)
