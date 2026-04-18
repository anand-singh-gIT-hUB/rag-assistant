"""
app/llm/factory.py
"""
from functools import lru_cache

from typing import Optional

from app.core.config import Settings, get_settings
from app.llm.base import LLMBase


@lru_cache(maxsize=1)
def _get_llm() -> LLMBase:
    settings = get_settings()
    if settings.llm_provider == "openai":
        from app.llm.openai_llm import OpenAILLM
        return OpenAILLM(api_key=settings.openai_api_key, model=settings.openai_llm_model)
    elif settings.llm_provider == "ollama":
        from app.llm.ollama_llm import OllamaLLM
        return OllamaLLM(base_url=settings.ollama_base_url, model=settings.ollama_model)
    else:
        raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")


def get_llm(settings: Optional[Settings] = None) -> LLMBase:
    """Public wrapper to get the cached LLM."""
    return _get_llm()
