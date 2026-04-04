"""
app/llm/ollama_llm.py
──────────────────────
Ollama local LLM via its REST API.
Uses a persistent httpx.Client for connection reuse,
and caps generation length via num_predict for speed.
"""
import json
import httpx

from app.llm.base import LLMBase
from app.core.exceptions import LLMError


class OllamaLLM(LLMBase):
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3") -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._client = httpx.Client(timeout=900)

    @property
    def model_name(self) -> str:
        return self._model

    def _build_payload(self, system_prompt: str, user_prompt: str, stream: bool) -> dict:
        return {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": stream,
            "options": {
                "num_predict": 120,
                "temperature": 0.2,
                "top_p": 0.9,
            },
        }

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        payload = self._build_payload(system_prompt, user_prompt, stream=False)
        try:
            r = self._client.post(f"{self._base_url}/api/chat", json=payload)
            r.raise_for_status()
            return r.json()["message"]["content"]
        except Exception as e:
            raise LLMError(f"Ollama completion failed: {e}") from e

    def stream_complete(self, system_prompt: str, user_prompt: str):
        payload = self._build_payload(system_prompt, user_prompt, stream=True)
        try:
            with self._client.stream("POST", f"{self._base_url}/api/chat", json=payload) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    if "message" in chunk and "content" in chunk["message"]:
                        yield chunk["message"]["content"]
        except Exception as e:
            raise LLMError(f"Ollama stream failed: {e}") from e
