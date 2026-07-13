"""ICD-10 codes prediction pipeline: extract -> retrieve -> verify

Stage 1 (LLM): separate raw transcript into patient/doctor dialog, then
    extract diagnoses + evidence from dialog
Stage 2 (retrieval): recall candidate ICD-10-CM codes from the official
    code table (never let the LLM invent code strings)
Stage 3 (LLM): verify each candidate against the evidence and select
    final codes with confidence + evidence spans
"""

import json
from dataclasses import dataclass, field

from .parsing import extract_json
from .prompts import DIALOG_PROMPT, EXTRACT_PROMPT, VERIFY_PROMPT, CONFIDENCE_PROMPT
from .providers.base import LLMProvider
from .retrieval import ICD10Index


@dataclass
class CodePrediction:
    code: str
    description: str
    confidence: float
    evidence: str


@dataclass
class PipelineResult:
    dialog: str = field(default_factory=str)
    problem_list: list[dict] = field(default_factory=list)
    candidates: list[dict] = field(default_factory=list)
    predictions: list[CodePrediction] = field(default_factory=list)
    raw: dict = field(default_factory=dict)

    def codes(self) -> list[str]:
        return [p.code for p in self.predictions]


class ICD10Pipeline:
    def __init__(self, provider: LLMProvider, index: ICD10Index, top_k: int = 10):
        self.provider = provider
        self.index = index
        self.top_k = top_k

    def separate(self, transcript: str) -> str:
        """Stage 0: Separate raw transcript into patient/doctor dialog."""
        out = self.provider.complete(DIALOG_PROMPT.format(transcript=transcript))
        return out

    def extract(self, dialog: str) -> list[dict]:
        """Stage 1: Extract diagnoses + evidence from dialog."""
        out = self.provider.complete(EXTRACT_PROMPT.format(transcript=dialog))
        data = extract_json(out) or {}
        return data.get("problem_list", [])

    def retrieve(self, problem_list: list[dict]) -> list[dict]:
        """Stage 2: BM25 recall over the official ICD-10-CM table."""
        candidates: dict[str, dict] = {}
        for problem in problem_list:
            term = problem.get("normalized_term") or problem.get("term", "")
            status = problem.get("status", "")
            attributes = problem.get("attributes", {})
            acuity = attributes.get("acuity", "")
            laterality = attributes.get("laterality", "")
            query = f"{term} {status} {acuity} {laterality}".strip()
            for code, desc, score in self.index.search(query, k=self.top_k):
                if code not in candidates or score > candidates[code]["score"]:
                    candidates[code] = {
                        "code": code,
                        "description": desc,
                        "score": score,
                        "matched_diagnosis": problem.get("term", ""),
                    }
        return sorted(candidates.values(), key=lambda c: -c["score"])

    def verify(self, transcript: str, problem_list: list[dict], candidates: list[dict]) -> list[CodePrediction]:
        """Stage 3: LLM selects only candidates supported by the transcript."""
        cand_block = "\n".join(f"- {c['code']}: {c['description']}" for c in candidates)
        diag_block = json.dumps(problem_list, ensure_ascii=False, indent=2)
        out = self.provider.complete(
            VERIFY_PROMPT.format(transcript=transcript, diagnoses=diag_block, candidates=cand_block)
        )
        data = extract_json(out) or {}
        valid_codes = {c["code"]: c["description"] for c in candidates}
        preds = []
        for item in data.get("selected_codes", []):
            code = str(item.get("code", "")).strip().upper()
            if code not in valid_codes:  # safety guard: no hallucinated codes
                continue
            preds.append(
                CodePrediction(
                    code=code,
                    description=valid_codes[code],
                    confidence=self.compute_confidence(transcript, problem_list, item),  # float(item.get("confidence", 0.0)),
                    evidence=str(item.get("evidence", "")),
                )
            )
        return preds

    def compute_confidence(self, transcript: str, problem_list: list[dict], code_item: dict) -> float:
        conf = self.provider.confidence(
            CONFIDENCE_PROMPT.format(transcript=transcript, diagnoses=problem_list, candidate=code_item)
        )
        print(f"Code: {code_item['code']} | conf: {conf}")
        return conf

    def run(self, transcript: str) -> PipelineResult:
        dialog = self.separate(transcript)
        problem_list = self.extract(dialog)
        candidates = self.retrieve(problem_list)
        predictions = self.verify(dialog, problem_list, candidates) if candidates else []
        return PipelineResult(dialog=dialog, problem_list=problem_list, candidates=candidates, predictions=predictions)
