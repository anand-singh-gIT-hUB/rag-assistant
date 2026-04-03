"""
app/ingestion/parser.py
────────────────────────
Dispatcher: selects the correct loader based on file extension.
"""
from pathlib import Path

from app.ingestion.loaders.base import BaseLoader, PageContent
from app.ingestion.loaders.pdf_loader import PDFLoader
from app.ingestion.loaders.docx_loader import DocxLoader
from app.ingestion.loaders.txt_loader import TxtLoader
from app.ingestion.loaders.markdown_loader import MarkdownLoader
from app.core.exceptions import UnsupportedFileTypeError

_LOADERS: dict[str, BaseLoader] = {
    ".pdf": PDFLoader(),
    ".docx": DocxLoader(),
    ".txt": TxtLoader(),
    ".md": MarkdownLoader(),
    ".markdown": MarkdownLoader(),
}


def parse_file(path: Path) -> list[PageContent]:
    """Load and return pages from *path* using the matching loader."""
    ext = path.suffix.lower()
    loader = _LOADERS.get(ext)
    if loader is None:
        raise UnsupportedFileTypeError(
            f"No loader for extension '{ext}'",
            detail=f"Supported: {list(_LOADERS.keys())}",
        )
    return loader.load(path)
