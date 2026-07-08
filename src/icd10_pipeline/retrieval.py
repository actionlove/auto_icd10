"""BM25 retrieval over the ICD-10-CM code table.

The demo ships with a small sample table in data/icd10cm_codes_sample.csv.
For real use, download the full table (free) from CMS:
https://www.cms.gov/medicare/coding-billing/icd-10-codes
and point ICD10Index at it. Swap BM25 for an embedding index (e.g.
sentence-transformers + FAISS) in production; the interface stays the same.
"""

import csv
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class ICD10Index:
    def __init__(self, code_table_csv: str | Path):
        self.codes: list[str] = []
        self.descriptions: list[str] = []
        with open(code_table_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.codes.append(row["code"].strip().upper())
                self.descriptions.append(row["description"].strip())
        if not self.codes:
            raise ValueError(f"No codes loaded from {code_table_csv}")
        self._bm25 = BM25Okapi([_tokenize(d) for d in self.descriptions])

    def __len__(self) -> int:
        return len(self.codes)

    def search(self, query: str, k: int = 10) -> list[tuple[str, str, float]]:
        """Return top-k (code, description, score) for a clinical term."""
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(self.codes[i], self.descriptions[i], float(scores[i])) for i in ranked if scores[i] > 0]
