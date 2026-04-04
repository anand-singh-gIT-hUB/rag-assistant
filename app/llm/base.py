"""
app/llm/base.py
"""
from abc import ABC, abstractmethod


class LLMBase(ABC):
    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        ...

    @abstractmethod
    def stream_complete(self, system_prompt: str, user_prompt: str):
        """Yields string chunks from the LLM."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...
