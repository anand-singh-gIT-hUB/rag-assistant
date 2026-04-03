"""
app/llm/factory.py
"""
from functools import lru_cache

from app.core.config import Settings
from app.llm.base import LLMBase


@lru_cache(maxsize=1)
def get_llm(settings: Settings) -> LLMBase:
    if settings.llm_provider == "openai":
        from app.llm.openai_llm import OpenAILLM
        return OpenAILLM(api_key=settings.openai_api_key, model=settings.openai_llm_model)
    elif settings.llm_provider == "ollama":
        from app.llm.ollama_llm import OllamaLLM
        return OllamaLLM(base_url=settings.ollama_base_url, model=settings.ollama_model)
    else:
        raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")
