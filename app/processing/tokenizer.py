"""
app/processing/tokenizer.py
────────────────────────────
Thin tiktoken helper used by the chunker for token-aware splitting.
"""
from functools import lru_cache

import tiktoken


@lru_cache(maxsize=4)
def get_encoding(model: str = "cl100k_base") -> tiktoken.Encoding:
    """Return a cached tiktoken encoding."""
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding(model)


def count_tokens(text: str, model: str = "cl100k_base") -> int:
    enc = get_encoding(model)
    return len(enc.encode(text))


def truncate_to_tokens(text: str, max_tokens: int, model: str = "cl100k_base") -> str:
    enc = get_encoding(model)
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])
