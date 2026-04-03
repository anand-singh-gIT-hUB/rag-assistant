"""
app/ingestion/cleaner.py
─────────────────────────
Post-processing of raw extracted text before chunking.
"""
from app.ingestion.loaders.base import PageContent
from app.utils.text_utils import clean_text


def clean_pages(pages: list[PageContent]) -> list[PageContent]:
    """Return cleaned copies of page content objects."""
    cleaned: list[PageContent] = []
    for page in pages:
        text = clean_text(page.text)
        if text:
            cleaned.append(
                PageContent(
                    text=text,
                    page_number=page.page_number,
                    section_title=page.section_title,
                )
            )
    return cleaned
