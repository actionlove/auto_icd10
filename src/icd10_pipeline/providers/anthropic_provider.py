"""Claude API provider. Docs: https://docs.claude.com/en/api/overview"""

import os
from .base import LLMProvider


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        import anthropic  # lazy import so the app runs without this SDK installed

        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    def complete(self, prompt: str, max_tokens: int = 2000) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")
