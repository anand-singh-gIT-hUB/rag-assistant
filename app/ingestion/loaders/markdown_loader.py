"""
app/ingestion/loaders/markdown_loader.py
"""
import re
from pathlib import Path

from app.ingestion.loaders.base import BaseLoader, PageContent
from app.core.exceptions import DocumentProcessingError


class MarkdownLoader(BaseLoader):
    supported_extensions = {".md", ".markdown"}

    def load(self, path: Path) -> list[PageContent]:
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
            sections = re.split(r"(?m)^(#{1,3} .+)$", raw)
            pages: list[PageContent] = []
            heading: str | None = None

            for part in sections:
                if re.match(r"^#{1,3} ", part):
                    heading = part.lstrip("#").strip()
                elif part.strip():
                    pages.append(PageContent(text=part.strip(), section_title=heading))

            return pages or [PageContent(text=raw)]
        except Exception as e:
            raise DocumentProcessingError(
                f"Failed to parse Markdown: {path.name}", detail=str(e)
            ) from e
