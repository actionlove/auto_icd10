"""Gradio demo: ICD-10 code prediction from a doctor-patient transcript.

Run locally:
```
$ python app.py
```

On HF Spaces: this file is the Space entrypoint (sdk: gradio).
"""

import json
from pathlib import Path

import pandas as pd
import gradio as gr
from dotenv import load_dotenv

from icd10_pipeline.pipeline import ICD10Pipeline  # noqa: E402
from icd10_pipeline.providers import PROVIDERS, get_provider  # noqa: E402
from icd10_pipeline.retrieval import ICD10Index  # noqa: E402

load_dotenv()

CODE_TABLE = Path(__file__).parent / "data" / "icd10cm_codes_full.csv"
INDEX = ICD10Index(CODE_TABLE)
_PROVIDER_CACHE: dict[str, object] = {}

EXAMPLE = (
    "Good morning, what brings you in today? "
    "I've had this cough for five days, and a sore throat. "
    "Any fever? "
    "Around 100.5 last night. "
    "Throat is red, lungs are clear. Looks like an upper respiratory "
    "infection. Your blood pressure is also high again, 152/94 — you're on "
    "lisinopril for your hypertension, so let's increase the dose."
)


DEV_SAMPLE_SET_PATH = Path(__file__).parent / "data" / "Test_Project_ICD10_Dataset.csv"
DEV_SAMPLE_SET = pd.read_csv(DEV_SAMPLE_SET_PATH)
NUM_DEMO_SAMPLES = 10
DEV_SAMPLE_LIST = DEV_SAMPLE_SET["transcript"][:NUM_DEMO_SAMPLES].tolist()
NEXT_DEMO_SAMPLE_IDX = 0
def fill_with_next_sample() -> str:
    global NEXT_DEMO_SAMPLE_IDX
    if NEXT_DEMO_SAMPLE_IDX >= len(DEV_SAMPLE_LIST):
        NEXT_DEMO_SAMPLE_IDX = 0
    sample = DEV_SAMPLE_LIST[NEXT_DEMO_SAMPLE_IDX]
    NEXT_DEMO_SAMPLE_IDX += 1
    return sample


def _get_pipeline(provider_name: str, model_override: str) -> ICD10Pipeline:
    key = f"{provider_name}:{model_override or 'default'}"
    if key not in _PROVIDER_CACHE:
        _PROVIDER_CACHE[key] = get_provider(provider_name, model=model_override or None)
    return ICD10Pipeline(_PROVIDER_CACHE[key], INDEX)


def _parse_step_input(text: str, key: str) -> list:
    """Parse an (possibly hand-edited) intermediate JSON textbox.

    Accepts either a bare JSON list, or an object like {"<key>": [...]}.
    Raises ValueError with a human-readable message on bad input.
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in '{key}' box (line {e.lineno}, col {e.colno}): {e.msg}")
    if isinstance(data, dict):
        data = data.get(key)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list (or an object with a '{key}' list).")
    return data


def _fmt(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# One-click flow
# ---------------------------------------------------------------------------

def predict(transcript: str, provider_name: str, model_override: str):
    """E2E ICD-10-CM code prediction."""
    if not transcript.strip():
        return [], "Please paste a transcript.", "{}"
    pipeline = _get_pipeline(provider_name, model_override)
    try:
        result = pipeline.run(transcript)
    except Exception as e:  # surface API errors in the UI
        return [], f"Error: {type(e).__name__}: {e}", "{}"

    table = [
        [p.code, p.description, f"{p.confidence:.2f}", p.evidence]
        for p in result.predictions
    ]
    summary = (
        f"Extracted {len(result.problem_list)} diagnoses -> "
        f"{len(result.candidates)} candidate codes -> "
        f"{len(result.predictions)} final codes."
    )
    debug = _fmt({"dialog": result.dialog, "diagnoses": result.problem_list, "candidates": result.candidates[:20]})
    return table, summary, debug


# ---------------------------------------------------------------------------
# Stepwise debugging flow
# ---------------------------------------------------------------------------

def run_step1(transcript: str, provider_name: str, model_override: str):
    """Separate only (1 LLM call). Output goes to the editable Step 1 box."""
    if not transcript.strip():
        return "", "Please paste a transcript first."
    try:
        dialog = _get_pipeline(provider_name, model_override).separate(transcript)
    except Exception as e:
        return "", f"❌ Step 1 error: {type(e).__name__}: {e}"
    status = (
        f"✅ Step 1 (separate, provider={provider_name}): {len(dialog)} dialog. "
        "Edit the text dialog below if needed, then run Step 2."
    )
    if not dialog:
        status = "⚠️ Step 1 returned no diagnoses — check the transcript or the dialog prompt."
    return dialog, status


def run_step2(dialog: str, provider_name: str, model_override: str):
    """Extract only (1 LLM call). Output goes to the editable Step 2 box."""
    if not dialog.strip():
        return "", "Please paste a dialog first."
    try:
        diagnoses = _get_pipeline(provider_name, model_override).extract(dialog)
    except Exception as e:
        return "", f"❌ Step 2 error: {type(e).__name__}: {e}"
    status = (
        f"✅ Step 2 (extract, provider={provider_name}): {len(diagnoses)} diagnoses. "
        "Edit the JSON below if needed, then run Step 3."
    )
    if not diagnoses:
        status = "⚠️ Step 2 returned no diagnoses — check the dialog or the extract prompt."
    return _fmt(diagnoses), status


def run_step3(step2_output: str):
    """Retrieve only (no LLM call — pure BM25). Reads the Step 1 box."""
    if not step2_output.strip():
        return "", "Step 1 output is empty — run Step 1 first (or paste diagnoses JSON)."
    try:
        diagnoses = _parse_step_input(step2_output, "diagnoses")
        candidates = ICD10Pipeline(provider=None, index=INDEX).retrieve(diagnoses)
    except ValueError as e:
        return "", f"❌ Step 3 input error: {e}"
    except Exception as e:
        return "", f"❌ Step 3 error: {type(e).__name__}: {e}"
    status = (
        f"✅ Step 3 (retrieve, no API cost): {len(candidates)} candidate codes "
        f"from {len(diagnoses)} diagnoses over {len(INDEX):,}-code table. "
        "Edit below (add/remove candidates), then run Step 3."
    )
    if not candidates:
        status = "⚠️ Step 3 found no candidates — check normalized_term wording vs the code table."
    return _fmt(candidates), status


def run_step4(step1_output: str, step2_output: str, step3_output: str,
              provider_name: str, model_override: str):
    """Verify only (1 LLM call). Reads transcript + Step 2 + Step 3 boxes."""
    if not step3_output.strip():
        return [], "Step 3 output is empty — run Step 3 first (or paste candidates JSON)."
    try:
        diagnoses = _parse_step_input(step2_output, "diagnoses") if step2_output.strip() else []
        candidates = _parse_step_input(step3_output, "candidates")
        for c in candidates:  # validate minimal shape early, with a clear message
            if "code" not in c or "description" not in c:
                raise ValueError("Each candidate needs at least 'code' and 'description' fields.")
        preds = _get_pipeline(provider_name, model_override).verify(step1_output, diagnoses, candidates)
    except ValueError as e:
        return [], f"❌ Step 4 input error: {e}"
    except Exception as e:
        return [], f"❌ Step 4 error: {type(e).__name__}: {e}"
    table = [[p.code, p.description, f"{p.confidence:.2f}", p.evidence] for p in preds]
    return table, (
        f"✅ Step 4 (verify, provider={provider_name}): "
        f"{len(preds)} of {len(candidates)} candidates selected."
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

RESULT_HEADERS = ["Code", "Description", "Confidence", "Evidence"]
with gr.Blocks(title="ICD-10 Prediction Demo") as demo:
    gr.Markdown(
        "# ICD-10 Code Prediction from raw doctor/patient Transcripts\n"
        "Pipeline: **extract → retrieve (BM25 over official code table) → verify**. "
        "Codes are never generated from LLM memory — only selected from retrieved candidates.\n\n"
        "⚠️ Demo only. Not for clinical or billing use without certified-coder review."
    )
    with gr.Row():
        provider = gr.Dropdown(choices=list(PROVIDERS), value="openai", label="Provider")
        model = gr.Textbox(label="Model override (optional)", placeholder="e.g. gpt-4o-mini")

    btn_next = gr.Button("Next demo sample", variant="primary")
    transcript = gr.Textbox(lines=12, label="Doctor-patient raw transcript", value=EXAMPLE)
    btn_next.click(fn=fill_with_next_sample, outputs=transcript)

    with gr.Tab("One-click"):
        btn_pred = gr.Button("Predict ICD-10 codes", variant="primary")
        summary = gr.Markdown()
        output = gr.Dataframe(headers=RESULT_HEADERS, label="Predicted codes")
        with gr.Accordion("Pipeline debug (extraction + candidates)", open=False):
            debug = gr.Code(language="json")
        btn_pred.click(predict, inputs=[transcript, provider, model], outputs=[output, summary, debug])

    with gr.Tab("Stepwise debugging"):
        gr.Markdown(
            "Run one stage at a time. Both output boxes are **editable** — you can modify each stage's output "
            "to test how downstream stages react. Step 2 makes no API call."
        )
        step1_btn = gr.Button("Step 1 — Speaker diarization", variant="primary")
        step1_status = gr.Markdown()
        step1_out = gr.Textbox(label="Step 1 output — Dialog after separating speakers (editable)",
                               interactive=True)

        step2_btn = gr.Button("Step 2 — Extract diagnoses", variant="primary")
        step2_status = gr.Markdown()
        step2_out = gr.Textbox(label="Step 2 output — diagnoses JSON (editable)",
                               interactive=True)

        step3_btn = gr.Button("Step 3 — Retrieve candidate codes", variant="primary")
        step3_status = gr.Markdown()
        step3_out = gr.Textbox(label="Step 3 output — candidates JSON (editable)",
                               interactive=True)

        step4_btn = gr.Button("Step 4 — Verify & select final codes", variant="primary")
        step4_status = gr.Markdown()
        step4_out = gr.Dataframe(headers=RESULT_HEADERS, label="Final predicted codes")

        step1_btn.click(run_step1, inputs=[transcript, provider, model],
                        outputs=[step1_out, step1_status])
        step2_btn.click(run_step2, inputs=[step1_out, provider, model],
                        outputs=[step2_out, step2_status])
        step3_btn.click(run_step3, inputs=[step2_out],
                        outputs=[step3_out, step3_status])
        step4_btn.click(run_step4, inputs=[step1_out, step2_out, step3_out, provider, model],
                        outputs=[step4_out, step4_status])


if __name__ == "__main__":
    demo.launch()
