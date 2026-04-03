"""
app/ingestion/loaders/txt_loader.py
"""
from pathlib import Path

from app.ingestion.loaders.base import BaseLoader, PageContent
from app.core.exceptions import DocumentProcessingError


class TxtLoader(BaseLoader):
    supported_extensions = {".txt"}

    def load(self, path: Path) -> list[PageContent]:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            return [PageContent(text=text)]
        except Exception as e:
            raise DocumentProcessingError(
                f"Failed to read TXT: {path.name}", detail=str(e)
            ) from e
