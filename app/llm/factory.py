"""
app/llm/factory.py
"""
from functools import lru_cache

from app.core.config import get_settings
from app.llm.base import LLMBase


@lru_cache(maxsize=1)
def get_llm() -> LLMBase:
    """Zero-argument cached factory for the LLM. Strictly local Ollama."""
    settings = get_settings()
    if settings.llm_provider == "ollama":
        from app.llm.ollama_llm import OllamaLLM
        return OllamaLLM(base_url=settings.ollama_base_url, model=settings.ollama_model)
    else:
        raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}. Only 'ollama' is active.")
