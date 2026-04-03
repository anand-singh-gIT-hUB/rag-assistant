"""
app/llm/ollama_llm.py
──────────────────────
Ollama local LLM via its REST API.
"""
import httpx

from app.llm.base import LLMBase
from app.core.exceptions import LLMError


class OllamaLLM(LLMBase):
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3") -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        }
        try:
            r = httpx.post(
                f"{self._base_url}/api/chat",
                json=payload,
                timeout=900,
            )
            r.raise_for_status()
            return r.json()["message"]["content"]
        except Exception as e:
            raise LLMError(f"Ollama completion failed: {e}") from e
