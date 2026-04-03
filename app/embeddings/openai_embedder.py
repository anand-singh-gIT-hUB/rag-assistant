"""
app/embeddings/openai_embedder.py
"""
from functools import cached_property

from openai import OpenAI

from app.embeddings.base import EmbedderBase
from app.core.exceptions import EmbeddingError

_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedder(EmbedderBase):
    def __init__(self, api_key: str, model: str = "text-embedding-3-small") -> None:
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def embed_query(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = self._client.embeddings.create(input=texts, model=self._model)
            return [item.embedding for item in response.data]
        except Exception as e:
            raise EmbeddingError(f"OpenAI embedding failed: {e}") from e

    @cached_property
    def dimension(self) -> int:
        return _DIMENSIONS.get(self._model, 1536)
