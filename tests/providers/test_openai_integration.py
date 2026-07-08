"""Integration test: real OpenAI API call through OpenAIProvider.

Budget-conscious by design:
- Auto-SKIPS when OPENAI_API_KEY is not set (so `pytest tests/` stays free in CI).
- Uses gpt-4o-mini (cheapest mainline model) unless OPENAI_MODEL overrides it.
- Tiny prompt + max_tokens=10  ->  roughly ~30 tokens/test, i.e. a fraction of
  a cent per full run.

Run only this file:
    python -m pytest tests/providers/test_openai_integration.py -v

Force-skip even with a key present (e.g. in CI):
    SKIP_API_TESTS=1 python -m pytest tests/ -v
"""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent.parent

# Pick up OPENAI_API_KEY from .env if present (never commit .env)
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

requires_openai = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") or os.getenv("SKIP_API_TESTS") == "1",
    reason="OPENAI_API_KEY not set (or SKIP_API_TESTS=1) — skipping paid API test",
)

CHEAP_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


@pytest.fixture(scope="module")
def provider():
    """One provider (one client) shared by all tests in this module."""
    from icd10_pipeline.providers.openai_provider import OpenAIProvider

    return OpenAIProvider(model=CHEAP_MODEL)


@requires_openai
class TestOpenAIProviderLive:
    def test_complete_returns_text(self, provider):
        """Smallest possible round-trip: prompt in, non-empty text out."""
        out = provider.complete("Reply with exactly one word: OK", max_tokens=10)
        assert isinstance(out, str)
        assert out.strip(), "API returned empty text"
        assert "ok" in out.lower()

    def test_max_tokens_caps_output_length(self, provider):
        """Ask for a long answer but cap at 10 tokens — response must be short.

        Guards against the provider silently ignoring the max_tokens argument,
        which is exactly the bug that would blow up a real budget.
        """
        out = provider.complete("Count from 1 to 500, comma separated.", max_tokens=10)
        # 10 tokens is roughly <= 80 characters; use a generous bound
        assert len(out) < 120, f"max_tokens not respected, got {len(out)} chars"

    def test_json_instruction_parses(self, provider):
        """Mini version of what extract/verify need: strict-JSON output."""
        from icd10_pipeline.parsing import extract_json

        out = provider.complete(
            'Return ONLY this JSON, no markdown fences: {"status": "ok"}',
            max_tokens=15,
        )
        assert extract_json(out) == {"status": "ok"}


@requires_openai
def test_wrong_key_raises_auth_error(monkeypatch):
    """Invalid key must raise (not return garbage) so the UI can surface it."""
    import openai

    from icd10_pipeline.providers.openai_provider import OpenAIProvider

    bad = OpenAIProvider(model=CHEAP_MODEL, api_key="sk-invalid-key-000")
    with pytest.raises(openai.AuthenticationError):
        bad.complete("hi", max_tokens=5)
