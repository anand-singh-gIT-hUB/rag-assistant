"""
app/schemas/query.py
"""
from typing import Optional
from pydantic import BaseModel, Field


class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    file_name: str
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    excerpt: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: Optional[int] = Field(default=None, ge=1, le=50)
    doc_ids: Optional[list[str]] = Field(default=None, description="Filter to specific documents")
    rerank: Optional[bool] = Field(default=None, description="Override reranker toggle")
    stream: bool = Field(default=False, description="Whether to stream the response back")


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    model: str
    retrieval_top_k: int
    reranked: bool


class RetrieveRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: Optional[int] = Field(default=None, ge=1, le=50)
    doc_ids: Optional[list[str]] = None


class ChunkResult(BaseModel):
    chunk_id: str
    doc_id: str
    file_name: str
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    text: str
    score: float


class RetrieveResponse(BaseModel):
    question: str
    chunks: list[ChunkResult]
