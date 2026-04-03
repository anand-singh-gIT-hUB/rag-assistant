"""
app/utils/text_utils.py
"""
import re
import unicodedata


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces/newlines into single spaces."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def truncate(text: str, max_chars: int = 200) -> str:
    """Return text truncated to *max_chars* with ellipsis."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


def clean_text(text: str) -> str:
    """Remove null bytes and other control characters."""
    text = text.replace("\x00", "")
    # Remove other non-printable control chars (except \n \t)
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return normalize_whitespace(text)


def word_count(text: str) -> int:
    return len(text.split())
