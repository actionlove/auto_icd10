"""OpenAI API provider."""

import os
from .base import LLMProvider


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        from openai import OpenAI  # lazy import

        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def complete(self, prompt: str, max_tokens: int = 16384) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""
