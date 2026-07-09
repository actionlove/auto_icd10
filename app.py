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


def predict(transcript: str, provider_name: str, model_override: str):
    if not transcript.strip():
        return [], "Please paste a transcript.", "{}"
    key = f"{provider_name}:{model_override or 'default'}"
    if key not in _PROVIDER_CACHE:
        _PROVIDER_CACHE[key] = get_provider(provider_name, model=model_override or None)
    pipeline = ICD10Pipeline(_PROVIDER_CACHE[key], INDEX)
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
    debug = json.dumps(
        {"dialog": result.dialog, "diagnoses": result.problem_list, "candidates": result.candidates[:20]},
        indent=2,
        ensure_ascii=False,
    )
    return table, summary, debug


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

    transcript = gr.Textbox(lines=12, label="Doctor-patient raw transcript", value=EXAMPLE)

    with gr.Row():
        btn_next = gr.Button("Next demo sample", variant="primary")
        btn_pred = gr.Button("Predict ICD-10 codes", variant="primary")

    summary = gr.Markdown()
    output = gr.Dataframe(headers=["Code", "Description", "Confidence", "Evidence"], label="Predicted codes")
    with gr.Accordion("Pipeline debug (extraction + candidates)", open=False):
        debug = gr.Code(language="json")

    btn_next.click(fn=fill_with_next_sample, outputs=transcript)
    btn_pred.click(predict, inputs=[transcript, provider, model], outputs=[output, summary, debug])


if __name__ == "__main__":
    demo.launch()
