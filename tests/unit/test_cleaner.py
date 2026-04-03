"""
tests/unit/test_cleaner.py
"""
from app.ingestion.cleaner import clean_pages
from app.ingestion.loaders.base import PageContent


def test_strips_empty_pages():
    pages = [PageContent(text="   "), PageContent(text="hello")]
    result = clean_pages(pages)
    assert len(result) == 1
    assert result[0].text == "hello"


def test_normalises_whitespace():
    pages = [PageContent(text="hello   world\n\n\nfoo")]
    result = clean_pages(pages)
    assert "  " not in result[0].text


def test_preserves_metadata():
    pages = [PageContent(text="content", page_number=3, section_title="Intro")]
    result = clean_pages(pages)
    assert result[0].page_number == 3
    assert result[0].section_title == "Intro"
