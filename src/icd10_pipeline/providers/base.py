"""Provider abstraction: one interface, three backends (Anthropic / OpenAI / local)."""

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    name: str = "base"

    @abstractmethod
    def complete(self, prompt: str, max_tokens: int = 2000) -> str:
        """Send a single-turn prompt, return the text completion."""
        raise NotImplementedError

    @abstractmethod
    def confidence(self, prompt: str, max_tokens: int = 2000) -> str:
        """Send a single-turn prompt, return yes/no logit to compute confidence."""
        raise NotImplementedError
