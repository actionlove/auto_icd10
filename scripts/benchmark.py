"""Benchmark the pipeline on a csv test set.

Each line: {"encounter_id": str, "age": int, "age_unit": str, "sex": str, "visit_reason": str, "reference_answer": str, "transcript": str}

Usage:
    python scripts/benchmark.py --provider openai --test-set data/Test_Project_ICD10_Dataset.csv --limit 10

Reports per-sample and micro-averaged precision / recall / F1 (exact code
match), plus category-level (first 3 chars) recall — useful because a
near-miss like E11.9 vs E11.65 matters differently for billing.
Writes per-sample results to results/benchmark_<provider>.json.
"""

import argparse
import csv
import json
import pandas as pd
from pathlib import Path
import sys
import time

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv

from icd10_pipeline.pipeline import ICD10Pipeline
from icd10_pipeline.providers import PROVIDERS, get_provider
from icd10_pipeline.retrieval import ICD10Index


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=list(PROVIDERS), default="anthropic")
    ap.add_argument("--model", default=None)
    ap.add_argument("--test-set", default=str(ROOT / "data" / "Test_Project_ICD10_Dataset.csv"))
    ap.add_argument("--code-table", default=str(ROOT / "data" / "icd10cm_codes_full.csv"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")

    samples = pd.read_csv(args.test_set)
    if args.limit and args.limit > 0 and args.limit < len(samples):
        samples = samples.iloc[:args.limit]

    pipeline = ICD10Pipeline(get_provider(args.provider, model=args.model), ICD10Index(args.code_table))

    rows, TP = [], 0
    FP = FN = cat_tp = cat_fn = 0
    t0 = time.time()

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)

    # Open csv file for debug output. With newline="" to prevent blank rows
    output_keys = samples.columns.tolist() + ["dialog", "problem_list", "candidates", "predictions"]
    debug_out = out_dir / f"debug_output_{args.provider}.csv"
    with open(debug_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_keys)
        writer.writeheader()

        for count, (index, s) in enumerate(samples.iterrows()):
            gold = {c.upper().split()[0] for c in s["reference_answer"].split("\n") if c.strip()}
            try:
                results = pipeline.run(s["transcript"])
                pred = set(results.codes())
                err = None
            except Exception as e:
                pred, err = set(), f"{type(e).__name__}: {e}"

            # Dump debug output
            dump_data = {**s.to_dict(), "dialog": results.dialog,
                    "problem_list": json.dumps(results.problem_list),
                    "candidates": json.dumps(results.candidates),
                    "predictions": json.dumps([p.__dict__ for p in results.predictions])}
            writer.writerow(dump_data)
            print(f"Finished sample: {count}")

            tp, fp, fn = len(pred & gold), len(pred - gold), len(gold - pred)
            TP, FP, FN = TP + tp, FP + fp, FN + fn
            gold_cat, pred_cat = {c[:3] for c in gold}, {c[:3] for c in pred}
            cat_tp += len(pred_cat & gold_cat)
            cat_fn += len(gold_cat - pred_cat)
            p, r, f1 = prf(tp, fp, fn)
            rows.append({"id": s["encounter_id"], "gold": sorted(gold), "pred": sorted(pred),
                        "precision": p, "recall": r, "f1": f1, "error": err})
            print(f"[{s['encounter_id']}] P={p:.2f} R={r:.2f} F1={f1:.2f}  pred={sorted(pred)}  gold={sorted(gold)}"
                + (f"  ERROR={err}" if err else ""))

    P, R, F1 = prf(TP, FP, FN)
    cat_recall = cat_tp / (cat_tp + cat_fn) if cat_tp + cat_fn else 0.0
    elapsed = time.time() - t0
    print("\n===== MICRO-AVERAGED (exact code match) =====")
    print(f"Precision: {P:.3f}   Recall: {R:.3f}   F1: {F1:.3f}")
    print(f"Category-level (3-char) recall: {cat_recall:.3f}")
    print(f"Samples: {len(samples)}   Time: {elapsed:.1f}s ({elapsed/len(samples):.1f}s/sample)")

    out = out_dir / f"benchmark_{args.provider}.json"
    out.write_text(json.dumps({
        "provider": args.provider, "model": args.model,
        "micro": {"precision": P, "recall": R, "f1": F1},
        "category_recall": cat_recall, "samples": rows,
    }, indent=2))

    print(f"Per-sample results -> {out}")
    print(f"Debug results -> {debug_out}")


if __name__ == "__main__":
    main()
