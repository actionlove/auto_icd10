# CLAUDE.md — Project guide for Claude Code

## What this project is
`auto_icd10` pipeline predicting ICD-10-CM codes from doctor-patient visit transcripts.
Three stages: **extract** (LLM) → **retrieve** (BM25 over official code table) →
**verify** (LLM selects only retrieved candidates, with evidence quotes).

## Hard invariants — never break these
1. **Never let an LLM emit an ICD code from memory.** All final codes must come
   from the retrieval candidate list (`pipeline.verify` filters against it).
   The test `tests/test_pipeline.py::test_end_to_end_with_mock` enforces this.
2. Every predicted code must carry an `evidence` quote from the transcript.
3. This is a demo: keep the "not for clinical/billing use" warning in the UI.
4. Never commit `.env` or any real patient data. Sample data in `data/` is synthetic.

## Layout
- `app.py` — Gradio UI (also the HF Spaces entrypoint)
- `src/icd10_pipeline/pipeline.py` — orchestrator
- `src/icd10_pipeline/providers/` — anthropic / openai / local backends (add new ones here, implement `LLMProvider.complete`)
- `src/icd10_pipeline/retrieval.py` — ICD10Index (BM25; swap for embeddings here)
- `src/icd10_pipeline/prompts.py` — all prompts live here, nowhere else
- `scripts/benchmark.py` — evaluation on JSONL test sets
- `data/` — sample code table + synthetic test set

## Commands
- Install: `pip install -r requirements.txt` (add `transformers torch accelerate` for local provider)
- Run app: `python app.py`
- Tests: `python -m pytest tests/ -v` (no API keys required — provider is mocked)
- Benchmark: `python scripts/benchmark.py --provider anthropic`

## Style
- Python 3.10+, type hints, no heavy frameworks.
- Lazy-import provider SDKs so the app runs with only the SDKs the user installed.
- Prompts return strict JSON; parsing goes through `parsing.extract_json` only.
