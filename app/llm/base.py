"""
app/llm/base.py
"""
from abc import ABC, abstractmethod


class LLMBase(ABC):
    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...
