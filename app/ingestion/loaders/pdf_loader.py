"""
app/ingestion/loaders/pdf_loader.py
─────────────────────────────────────
PDF loader using PyMuPDF (fitz).
"""
from pathlib import Path

import fitz  # PyMuPDF

from app.ingestion.loaders.base import BaseLoader, PageContent
from app.core.exceptions import DocumentProcessingError


class PDFLoader(BaseLoader):
    supported_extensions = {".pdf"}

    def load(self, path: Path) -> list[PageContent]:
        pages: list[PageContent] = []
        try:
            doc = fitz.open(str(path))
            for i, page in enumerate(doc, start=1):
                text = page.get_text("text")
                if text.strip():
                    pages.append(PageContent(text=text, page_number=i))
            doc.close()
        except Exception as e:
            raise DocumentProcessingError(
                f"Failed to parse PDF: {path.name}", detail=str(e)
            ) from e
        return pages
