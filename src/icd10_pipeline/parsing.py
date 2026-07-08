"""Robust extraction of a JSON object from an LLM response."""

import json
import re

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> dict | None:
    """Best-effort parse of the first JSON object found in `text`."""
    if not text:
        return None

    # 1. try fenced block
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1)

    # 2. try direct parse
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass

    # 3. try first {...} span (greedy from first '{' to last '}')
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None

    return None
