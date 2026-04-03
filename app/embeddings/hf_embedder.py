"""
app/embeddings/hf_embedder.py
──────────────────────────────
HuggingFace sentence-transformers embedder (no API key required).
"""
from functools import cached_property

from app.embeddings.base import EmbedderBase
from app.core.exceptions import EmbeddingError


class HuggingFaceEmbedder(EmbedderBase):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model = None  # lazy-load

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self._model_name)
            except Exception as e:
                raise EmbeddingError(f"Failed to load HF model: {e}") from e
        return self._model

    def embed_query(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            model = self._get_model()
            vectors = model.encode(texts, convert_to_numpy=True)
            return [v.tolist() for v in vectors]
        except Exception as e:
            raise EmbeddingError(f"HF embedding failed: {e}") from e

    @cached_property
    def dimension(self) -> int:
        model = self._get_model()
        return model.get_sentence_embedding_dimension()
