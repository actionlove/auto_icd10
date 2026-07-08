"""Unit tests for ICD10Pipeline.extract() / .retrieve() / .verify()

All tests can run WITHOUT API keys: the LLM provider is replaced by DummyProvider,
which returns canned responses. Retrieval uses the real BM25 index over the
sample code table, so retrieve() behavior is tested against real ranking.

Run: python -m pytest tests/test_pipeline_units.py -v
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from icd10_pipeline.pipeline import ICD10Pipeline
from icd10_pipeline.providers.base import LLMProvider
from icd10_pipeline.retrieval import ICD10Index

TABLE = Path(__file__).parent.parent / "data" / "icd10cm_codes_sample.csv"


class DummyProvider(LLMProvider):
    """Returns pre-scripted responses in order; records every prompt it receives."""

    name = "dummy"

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError("DummyProvider ran out of scripted responses")
        return self.responses.pop(0)


@pytest.fixture(scope="module")
def index() -> ICD10Index:
    return ICD10Index(TABLE)


def make_pipeline(responses: list[str], index: ICD10Index, top_k: int = 10) -> ICD10Pipeline:
    return ICD10Pipeline(DummyProvider(responses), index, top_k=top_k)


# ---------------------------------------------------------------------------
# extract()
# ---------------------------------------------------------------------------

class TestExtract:
    SEP_TRANSCRIPT = "Patient: I am feeling dizzy today. Doctor: BP is high again."
    DIAG = {
        "subjective": "Patient reported feeling dizzy",
        "objective": "BP: 150/90",
        "assessment": "BP is high",
        "plan": "medication",
        "problem_list": [
            {
                "term": "high blood pressure",
                "normalized_term": "essential hypertension",
                "status": "chronic_active",
                "evidence": "blood pressure is running high",
                "attributes": {
                    "acuity": "chronic",
                    "laterality": "null",
                    "severity": "mild",
                    "causal_link": "null"
                },
                "addressed_this_visit": True
            }
        ],
        "ambiguities": []
    }

    def test_parses_clean_json(self, index):
        pipe = make_pipeline([json.dumps(self.DIAG)], index)
        diagnoses = pipe.extract(self.SEP_TRANSCRIPT)
        assert len(diagnoses) == 1
        assert diagnoses[0]["normalized_term"] == "essential hypertension"

    def test_parses_fenced_json(self, index):
        """LLMs often wrap JSON in markdown fences despite instructions."""
        fenced = "```json\n" + json.dumps(self.DIAG) + "\n```"
        pipe = make_pipeline([fenced], index)
        assert len(pipe.extract(self.SEP_TRANSCRIPT)) == 1

    def test_parses_json_with_preamble(self, index):
        chatty = "Sure, here is the extraction:\n" + json.dumps(self.DIAG)
        pipe = make_pipeline([chatty], index)
        assert len(pipe.extract(self.SEP_TRANSCRIPT)) == 1

    def test_malformed_json_returns_empty_list(self, index):
        pipe = make_pipeline(["I could not process this transcript."], index)
        assert pipe.extract(self.SEP_TRANSCRIPT) == []

    def test_missing_diagnoses_key_returns_empty_list(self, index):
        pipe = make_pipeline([json.dumps({"something_else": 1})], index)
        assert pipe.extract(self.SEP_TRANSCRIPT) == []

    def test_transcript_is_injected_into_prompt(self, index):
        provider = DummyProvider([json.dumps({})])
        pipe = ICD10Pipeline(provider, index)
        pipe.extract("UNIQUE_MARKER_12345 patient text")
        assert len(provider.prompts) == 1
        assert "UNIQUE_MARKER_12345" in provider.prompts[0]


# ---------------------------------------------------------------------------
# retrieve()  (no LLM involved — uses real BM25 index)
# ---------------------------------------------------------------------------

class TestRetrieve:
    def _diag(self, term: str, normalized: str | None = None) -> dict:
        d = {"term": term}
        if normalized is not None:
            d["normalized_term"] = normalized
        return d

    def test_finds_expected_code(self, index):
        pipe = make_pipeline([], index)
        cands = pipe.retrieve([self._diag("high blood pressure", "essential hypertension")])
        assert "I10" in [c["code"] for c in cands]

    def test_prefers_normalized_term_over_term(self, index):
        """Colloquial term alone should not drive retrieval when normalized exists."""
        pipe = make_pipeline([], index)
        cands = pipe.retrieve([self._diag("sugar problems", "type 2 diabetes mellitus with hyperglycemia")])
        codes = [c["code"] for c in cands]
        assert "E11.65" in codes

    def test_falls_back_to_term_when_no_normalized(self, index):
        pipe = make_pipeline([], index)
        cands = pipe.retrieve([self._diag("essential hypertension")])
        assert "I10" in [c["code"] for c in cands]

    def test_deduplicates_across_diagnoses(self, index):
        """Two diagnoses hitting the same code must yield one candidate entry."""
        pipe = make_pipeline([], index)
        cands = pipe.retrieve([
            self._diag("hypertension", "essential hypertension"),
            self._diag("high blood pressure", "primary hypertension"),
        ])
        codes = [c["code"] for c in cands]
        assert codes.count("I10") == 1

    def test_sorted_by_score_descending(self, index):
        pipe = make_pipeline([], index)
        cands = pipe.retrieve([
            self._diag("hypertension", "essential hypertension"),
            self._diag("reflux", "gastro-esophageal reflux disease"),
        ])
        scores = [c["score"] for c in cands]
        assert scores == sorted(scores, reverse=True)

    def test_empty_diagnoses_returns_empty(self, index):
        pipe = make_pipeline([], index)
        assert pipe.retrieve([]) == []

    def test_unmatchable_term_returns_empty(self, index):
        pipe = make_pipeline([], index)
        assert pipe.retrieve([self._diag("zzzzqqq nonsense")]) == []

    def test_respects_top_k(self, index):
        pipe_small = make_pipeline([], index, top_k=2)
        cands = pipe_small.retrieve([self._diag("unspecified", "unspecified")])
        # 'unspecified' matches many rows in the table; per-diagnosis recall capped at 2
        assert len(cands) <= 2

    def test_candidate_shape(self, index):
        pipe = make_pipeline([], index)
        cands = pipe.retrieve([self._diag("hypertension", "essential hypertension")])
        assert cands, "expected at least one candidate"
        assert {"code", "description", "score", "matched_diagnosis"} <= set(cands[0])


# ---------------------------------------------------------------------------
# verify()
# ---------------------------------------------------------------------------

class TestVerify:
    TRANSCRIPT = "Doctor: your blood pressure is running high again."
    DIAGNOSES = [{"term": "high blood pressure", "normalized_term": "essential hypertension"}]
    CANDIDATES = [
        {"code": "I10", "description": "Essential (primary) hypertension", "score": 9.0,
         "matched_diagnosis": "high blood pressure"},
        {"code": "E11.9", "description": "Type 2 diabetes mellitus without complications", "score": 1.0,
         "matched_diagnosis": "high blood pressure"},
    ]

    def _run(self, index, selected: list[dict]):
        pipe = make_pipeline([json.dumps({"selected_codes": selected})], index)
        return pipe.verify(self.TRANSCRIPT, self.DIAGNOSES, self.CANDIDATES)

    def test_selects_supported_candidate(self, index):
        preds = self._run(index, [
            {"code": "I10", "confidence": 0.95, "evidence": "blood pressure is running high"}])
        assert [p.code for p in preds] == ["I10"]
        assert preds[0].confidence == pytest.approx(0.95)
        assert preds[0].description == "Essential (primary) hypertension"
        assert "blood pressure" in preds[0].evidence

    def test_filters_hallucinated_code(self, index):
        """THE core invariant: a code outside the candidate list must never survive."""
        preds = self._run(index, [
            {"code": "I10", "confidence": 0.9, "evidence": "bp high"},
            {"code": "Z99.99", "confidence": 0.99, "evidence": "made up"},
        ])
        assert [p.code for p in preds] == ["I10"]

    def test_normalizes_code_case_and_whitespace(self, index):
        preds = self._run(index, [{"code": " i10 ", "confidence": 0.8, "evidence": "bp high"}])
        assert [p.code for p in preds] == ["I10"]

    def test_can_reject_all_candidates(self, index):
        """Verifier saying 'nothing is supported' must yield zero predictions."""
        preds = self._run(index, [])
        assert preds == []

    def test_malformed_llm_output_yields_empty(self, index):
        pipe = make_pipeline(["cannot help with that"], index)
        assert pipe.verify(self.TRANSCRIPT, self.DIAGNOSES, self.CANDIDATES) == []

    def test_missing_confidence_defaults_to_zero(self, index):
        preds = self._run(index, [{"code": "I10", "evidence": "bp high"}])
        assert preds[0].confidence == 0.0

    def test_candidates_and_transcript_in_prompt(self, index):
        provider = DummyProvider([json.dumps({"selected_codes": []})])
        pipe = ICD10Pipeline(provider, index)
        pipe.verify(self.TRANSCRIPT, self.DIAGNOSES, self.CANDIDATES)
        prompt = provider.prompts[0]
        assert "I10" in prompt and "E11.9" in prompt        # candidate block present
        assert "blood pressure is running high" in prompt    # transcript present
        assert "essential hypertension" in prompt            # diagnoses block present


# ---------------------------------------------------------------------------
# run()  (integration of the three stages)
# ---------------------------------------------------------------------------

class TestRunIntegration:
    def test_run_wires_stages_together(self, index):
        sep_resp = TestExtract.SEP_TRANSCRIPT
        extract_resp = json.dumps(TestExtract.DIAG)
        verify_resp = json.dumps({"selected_codes": [
            {"code": "I10", "confidence": 0.9, "evidence": "bp is high"}]})
        pipe = make_pipeline([sep_resp, extract_resp, verify_resp], index)
        result = pipe.run("I am feeling dizzy. bp is high.")
        assert result.codes() == ["I10"]
        assert len(result.problem_list) == 1
        assert any(c["code"] == "I10" for c in result.candidates)

    def test_run_skips_verify_when_no_candidates(self, index):
        """If retrieval finds nothing, verify must NOT be called (only 1 LLM call)."""
        provider = DummyProvider(["", json.dumps({})])
        pipe = ICD10Pipeline(provider, index)
        result = pipe.run("transcript")
        assert result.predictions == []
        assert len(provider.prompts) == 2
