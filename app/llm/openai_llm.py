"""
app/llm/openai_llm.py
"""
from openai import OpenAI

from app.llm.base import LLMBase
from app.core.exceptions import LLMError


class OpenAILLM(LLMBase):
    def __init__(self, api_key: str, model: str = "gpt-4.1-mini") -> None:
        self._client = OpenAI(api_key=api_key)
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            raise LLMError(f"OpenAI completion failed: {e}") from e
