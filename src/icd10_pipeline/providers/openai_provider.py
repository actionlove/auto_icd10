"""OpenAI API provider."""

import math
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

    def confidence(self, prompt: str, max_tokens: int = 16384) -> float:
        """Send a single-turn prompt, return yes/no logits to compute confidence."""
        # 1. Identify the exact Token IDs for your target words (using cl100k_base / o200k_base)
        YES_TOKEN_ID = 6763
        YES_SPACE_TOKEN_ID = 14531
        NO_TOKEN_ID = 1750
        NO_SPACE_TOKEN_ID = 860

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            logprobs=True,
            top_logprobs=2, # Ensure we capture both tokens in the response array
            max_tokens=1,   # Stop immediately after evaluating the first token
            logit_bias={
                YES_TOKEN_ID: 100,  # Flood the model's logits so it can only pick Yes...
                NO_TOKEN_ID: 100    # ...or No
            }
        )

        # 2. Extract the logprobs from the first generated token position
        top_candidates = response.choices[0].logprobs.content[0].top_logprobs

        # 3. Read the log probabilities directly
        logit_yes = logit_no = 0.0
        for candidate in top_candidates:
            print(f"Token: '{candidate.token}' | Logprob: {candidate.logprob}")
            if candidate.token == "yes":
                logit_yes = candidate.logprob
            elif candidate.token == "no":
                logit_no = candidate.logprob

        return math.exp(logit_yes) / (math.exp(logit_yes) + math.exp(logit_no))
