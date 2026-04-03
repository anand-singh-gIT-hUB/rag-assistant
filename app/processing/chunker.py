"""
app/processing/chunker.py
──────────────────────────
Token-aware recursive character text splitter.
Produces plain text chunks; caller is responsible for wrapping in ChunkMetadata.
"""
from __future__ import annotations

from app.processing.tokenizer import count_tokens


class RecursiveTextSplitter:
    """
    Split text recursively on a priority list of separators,
    respecting a token-budget per chunk.
    """

    SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> list[str]:
        """Return a list of text chunks."""
        chunks: list[str] = []
        self._split_recursive(text, self.SEPARATORS, chunks)
        return [c.strip() for c in chunks if c.strip()]

    # ── Internal ─────────────────────────────────────────────────────────────

    def _split_recursive(
        self, text: str, separators: list[str], output: list[str]
    ) -> None:
        if count_tokens(text) <= self.chunk_size:
            output.append(text)
            return

        sep = self._choose_separator(text, separators)
        parts = text.split(sep) if sep else list(text)

        current: list[str] = []
        current_tokens = 0

        for part in parts:
            part_tokens = count_tokens(part)
            if current_tokens + part_tokens > self.chunk_size and current:
                chunk_text = sep.join(current)
                output.append(chunk_text)
                # keep overlap
                overlap: list[str] = []
                overlap_tokens = 0
                for p in reversed(current):
                    overlap_tokens += count_tokens(p)
                    if overlap_tokens >= self.chunk_overlap:
                        break
                    overlap.insert(0, p)
                current = overlap
                current_tokens = sum(count_tokens(p) for p in current)

            current.append(part)
            current_tokens += part_tokens

        if current:
            output.append(sep.join(current))

    def _choose_separator(self, text: str, separators: list[str]) -> str:
        for sep in separators:
            if sep and sep in text:
                return sep
        return separators[-1]
