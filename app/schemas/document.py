"""
app/schemas/document.py
"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class DocumentResponse(BaseModel):
    doc_id: str
    file_name: str
    file_type: str
    file_size_bytes: int
    num_chunks: int
    uploaded_at: datetime
    status: str = "indexed"


class DocumentMeta(BaseModel):
    doc_id: str
    file_name: str
    file_type: str
    file_size_bytes: int
    num_chunks: int
    uploaded_at: datetime


class DocumentListResponse(BaseModel):
    documents: list[DocumentMeta]
    total: int
