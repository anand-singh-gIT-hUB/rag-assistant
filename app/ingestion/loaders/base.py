"""
app/ingestion/loaders/base.py
──────────────────────────────
Abstract base for all document loaders.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class PageContent:
    text: str
    page_number: Optional[int] = None
    section_title: Optional[str] = None


class BaseLoader(ABC):
    """Return a list of PageContent objects from a file path."""

    @abstractmethod
    def load(self, path: Path) -> list[PageContent]:
        ...

    @property
    @abstractmethod
    def supported_extensions(self) -> set[str]:
        ...
