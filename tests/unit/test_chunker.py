"""
tests/unit/test_chunker.py
"""
import pytest
from app.processing.chunker import RecursiveTextSplitter


def test_short_text_single_chunk():
    splitter = RecursiveTextSplitter(chunk_size=512, chunk_overlap=64)
    text = "Hello world. This is a short document."
    chunks = splitter.split(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_long_text_multiple_chunks():
    splitter = RecursiveTextSplitter(chunk_size=50, chunk_overlap=10)
    text = " ".join([f"word{i}" for i in range(200)])
    chunks = splitter.split(text)
    assert len(chunks) > 1


def test_empty_text_returns_empty():
    splitter = RecursiveTextSplitter()
    assert splitter.split("") == []
    assert splitter.split("   ") == []


def test_chunks_not_empty():
    splitter = RecursiveTextSplitter(chunk_size=100, chunk_overlap=20)
    text = "\n\n".join([f"Paragraph {i}: " + "content " * 30 for i in range(5)])
    chunks = splitter.split(text)
    assert all(c.strip() for c in chunks)
