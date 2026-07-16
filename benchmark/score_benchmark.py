#!/usr/bin/env python3
"""
score_benchmark.py
-------------------
Scores one or more prediction files against the gold benchmark, computing:
    BLEU, SacreBLEU, chrF, CER, WER, Exact Match

Requires:
    pip install sacrebleu jiwer

Usage:
    python score_benchmark.py \
        --predictions benchmark/results/predictions_run1.csv:run1 \
                       benchmark/results/predictions_run2.csv:run2 \
                       benchmark/results/predictions_mavkif_base.csv:mavkif_base \
        --output benchmark/results/benchmark_results.csv

Each --predictions entry is "path:model_label".
Outputs a scoreboard CSV (one row per model, overall + per-category) and
prints a human-readable table to stdout.
"""

import argparse
import csv
from collections import defaultdict

import sacrebleu
import jiwer


def load_predictions(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader)


def compute_metrics(rows):
    """rows: list of dicts with gold_roman_urdu, prediction, category"""
    refs = [r["gold_roman_urdu"].strip() for r in rows]
    hyps = [r["prediction"].strip() for r in rows]

    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    chrf = sacrebleu.corpus_chrf(hyps, [refs])

    # CER / WER via jiwer (aggregate across all sentences)
    # jiwer expects non-empty strings; guard against empty predictions.
    safe_hyps = [h if h else " " for h in hyps]
    safe_refs = [r if r else " " for r in refs]

    wer = jiwer.wer(safe_refs, safe_hyps)
    cer = jiwer.cer(safe_refs, safe_hyps)

    exact_matches = sum(1 for r, h in zip(refs, hyps) if r == h)
    exact_match_pct = 100.0 * exact_matches / len(refs) if refs else 0.0

    return {
        # NOTE: "BLEU" and "SacreBLEU" are the same metric computed the same way here.
        # SacreBLEU is a standardized, reproducible implementation of corpus BLEU —
        # there is no separate "BLEU" library distinct from it in standard practice.
        # We report one column (bleu) rather than two identical ones.
        "bleu": round(bleu.score, 2),
        "chrf": round(chrf.score, 2),
        "cer": round(cer * 100, 2),   # as percentage
        "wer": round(wer * 100, 2),   # as percentage
        "exact_match": round(exact_match_pct, 2),
        "n": len(refs),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions", nargs="+", required=True,
        help="One or more 'path:label' pairs, e.g. results/predictions_run2.csv:run2"
    )
    parser.add_argument("--output", default="benchmark/results/benchmark_results.csv")
    parser.add_argument("--by_category", action="store_true", help="Also break down scores by category")
    args = parser.parse_args()

    model_entries = []
    for entry in args.predictions:
        if ":" not in entry:
            raise ValueError(f"--predictions entries must be 'path:label', got: {entry}")
        path, label = entry.rsplit(":", 1)
        model_entries.append((path, label))

    all_results = []

    for path, label in model_entries:
        rows = load_predictions(path)
        missing_pred = [r for r in rows if not r.get("prediction", "").strip()]
        if missing_pred:
            print(f"[score_benchmark] WARNING: {label} has {len(missing_pred)} rows with empty predictions.")

        overall = compute_metrics(rows)
        overall["model"] = label
        overall["category"] = "ALL"
        all_results.append(overall)

        if args.by_category:
            by_cat = defaultdict(list)
            for r in rows:
                by_cat[r["category"]].append(r)
            for cat, cat_rows in sorted(by_cat.items()):
                cat_metrics = compute_metrics(cat_rows)
                cat_metrics["model"] = label
                cat_metrics["category"] = cat
                all_results.append(cat_metrics)

    fieldnames = ["model", "category", "n", "bleu", "chrf", "cer", "wer", "exact_match"]
    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n[score_benchmark] Results written to '{args.output}'\n")

    print(f"{'Model':<20}{'Category':<14}{'N':<6}{'BLEU':<8}{'chrF':<8}{'CER%':<8}{'WER%':<8}{'ExactMatch%':<12}")
    print("-" * 90)
    for r in all_results:
        print(
            f"{r['model']:<20}{r['category']:<14}{r['n']:<6}"
            f"{r['bleu']:<8}{r['chrf']:<8}{r['cer']:<8}{r['wer']:<8}{r['exact_match']:<12}"
        )


if __name__ == "__main__":
    main()
