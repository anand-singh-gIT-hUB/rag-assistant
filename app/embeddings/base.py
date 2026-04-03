"""
app/embeddings/base.py
"""
from abc import ABC, abstractmethod


class EmbedderBase(ABC):
    """Abstract embedder — all implementations must satisfy this contract."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding vector dimension."""
        ...
