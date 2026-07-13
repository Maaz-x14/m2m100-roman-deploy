"""
eval_layer2_openai.py
-----------------------
Layer 2 of the evaluation pipeline, using OpenAI instead of Gemini.

Sends Urdu input / Roman Urdu output pairs to an OpenAI model for
suspicious-pattern triage. The model is NOT asked to assert ground-truth
correctness (it has no more access to ground truth than we do) — it is
asked to flag rows matching specific named patterns, using a fixed rubric,
and to mark itself "uncertain" rather than guess. Final correctness
judgment stays with you in Layer 3 (manual review).

SCOPE — two modes:
  --scope all     : every row in layer1_results.csv is checked (full QA pass)
  --scope triage  : only flagged rows + a random sample of unflagged rows
                    (faster/cheaper, matches the original sampling plan)

REQUIRES
--------
pip install openai --break-system-packages
export OPENAI_API_KEY=your_key_here

USAGE
-----
python eval_layer2_openai.py \
    --layer1-csv layer1_results.csv \
    --output layer2_results.csv \
    --scope all \
    --batch-size 150 \
    --model gpt-5.4-mini

RESUMABLE — same pattern as Layer 1: writes results incrementally, keeps a
checkpoint file of processed message_ids, safe to re-run after interruption.

BEFORE RUNNING: use estimate_cost.py against your real layer1_results.csv
to see actual expected cost first. Do not skip this step.
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

try:
    from openai import OpenAI
except ImportError:
    print(
        "[ERROR] openai package not installed. Run:\n"
        "  pip install openai --break-system-packages",
        file=sys.stderr,
    )
    sys.exit(1)


# ── Structured output schema ──────────────────────────────────────────────────
# OpenAI's structured outputs require the ROOT schema to be a JSON object,
# not a bare array — a list[Model] response_format is rejected. Wrapping the
# list in a container object (BatchVerdicts) is required, not optional.

class RowVerdict(BaseModel):
    message_id: int
    verdict: Literal["clean", "suspicious", "uncertain"]
    issue_type: Optional[Literal[
        "script_leakage",
        "degenerate_output",
        "implausible_loanword",
        "likely_iss_uss_error",
        "phonetic_mismatch",
        "other",
    ]] = None
    reason: str


class BatchVerdicts(BaseModel):
    verdicts: list[RowVerdict]


# ── System prompt ─────────────────────────────────────────────────────────────
# Identical rubric/content to the Gemini version — provider-agnostic task
# definition, full context, and explicit anti-hallucination rules.

SYSTEM_PROMPT = """\
You are reviewing the output of a fine-tuned M2M100 LoRA machine learning \
model whose task is Urdu (Arabic script) -> Roman Urdu TRANSLITERATION \
(NOT translation). Roman Urdu is Urdu written phonetically in Latin \
characters, e.g. "وہ گھر پر نہیں ہے" -> "woh ghar par nahi hai".

CONTEXT ON THE MODEL (so you calibrate expectations correctly):
- The model is generally strong on conversational Urdu and on common \
tech/loanword vocabulary (exercise, exam, laptop, crash, delete, backup, \
password, interview, online, update, restart, corrupt, etc).
- KNOWN WEAKNESS #1 — iss/uss disambiguation: the Urdu word "اس" is \
orthographically identical for both "iss" (proximal/present, e.g. "this") \
and "uss" (distal/past, e.g. "that"). The model gets this right only \
about half the time. When you see "iss" or "uss" in the Roman output, \
check whether the Urdu input's tense/context (past-tense markers like \
تھا/تھی/تھے/گیا/گئی, or proximal/present markers like ابھی/یہاں/یہ) \
supports the choice made. Do not assume either is automatically wrong — \
only flag it if the context clearly points the other way.
- KNOWN WEAKNESS #2 — code-switching loanwords: English words embedded \
mid-Urdu-sentence sometimes hallucinate into unrelated English words \
(e.g. "message" wrongly becoming "barad", "spreadsheet" becoming \
"bread-sheet"). Watch for Roman output words that don't phonetically \
resemble anything in the Urdu input and don't look like a real word.
- This is a farmer-facing agricultural voice assistant. Expect domain \
vocabulary: crop names, fertilizers, pesticides, mandi (market) prices, \
weather, tehsil/district names, livestock. Do not flag technical \
agricultural terms as suspicious just because they are unfamiliar to you \
— only flag them if the Roman transliteration doesn't phonetically match \
the Urdu script.

YOUR TASK:
For each (Urdu input, Roman Urdu output) pair below, decide whether the \
output shows signs of being WRONG, using ONLY the specific patterns listed \
below. You are not being asked to certify the output as fully correct — \
you are flagging suspicious patterns for a human to review afterward. \
Assign exactly one verdict per row:

- "clean": no suspicious pattern detected. This does NOT mean you are \
certifying it as perfect Roman Urdu — only that nothing in your rubric \
below was triggered.
- "suspicious": one or more of the specific issues below is present. Set \
issue_type to the single best-matching category, and reason to a short \
(under 20 words) explanation of what specifically looks wrong.
- "uncertain": you cannot confidently judge this row (e.g. the Urdu input \
itself is garbled/unclear, or the sentence is ambiguous enough that \
either iss or uss could be correct, or you are not confident about a \
domain-specific term). Use this instead of guessing. Set reason to \
explain briefly why you're uncertain.
Check that every word/phrase present in the Urdu input has a corresponding transliteration in the Roman output, in the same order.

ISSUE TYPES (use exactly one per suspicious row):
1. script_leakage — the Roman output still contains Urdu/Arabic script \
characters instead of being fully transliterated.
2. degenerate_output — output is empty, or contains an obviously looping \
repeated phrase, or is drastically shorter than the input would warrant.
3. implausible_loanword — a word in the output doesn't phonetically \
resemble anything in the Urdu input and doesn't look like a real English \
loanword — likely a hallucination (see known weakness #2 above).
4. likely_iss_uss_error — the model chose "iss" or "uss" and the Urdu \
input's tense/context markers clearly point to the OTHER choice (see \
known weakness #1 above). Only flag this when you have a specific \
textual reason, not a guess.
5. phonetic_mismatch — some other word or phrase in the output does not \
phonetically match its Urdu source (not covered by the above categories).
6. other — a real issue that doesn't fit the above categories. Explain \
clearly in reason.
7. omitted_or_reordered — one or more words/clauses from the Urdu input \
are missing from the Roman output, or the output's word order doesn't \
match the Urdu input's word order (not just a stylistic reordering that \
still preserves all content).

CRITICAL RULES — DO NOT VIOLATE:
- Do NOT invent or assume information not present in the Urdu input. If \
you are not sure what a word means, say so via "uncertain" rather than \
guessing at a verdict.
- Do NOT flag purely stylistic differences (e.g. "kya" vs "kia" spelling \
variants) as suspicious — Roman Urdu has no single standardized spelling.
- Do NOT flag known agricultural/technical vocabulary as suspicious \
merely because it is domain-specific — only flag if the transliteration \
itself doesn't match the Urdu source phonetically.
- You MUST return exactly one verdict object per row in the input batch, \
using the exact message_id given. Do not skip any row. Do not add rows \
that were not in the input. Do not merge two rows into one verdict.
- If you are unsure whether something is an error, choose "uncertain" — \
never present a guess as a confident "suspicious" verdict.

You will be given a numbered batch of rows, each with a message_id, the \
Urdu input, and the Roman Urdu output. Return a JSON object with a \
"verdicts" array containing exactly one verdict object per row, in any \
order, each carrying its correct message_id.
Just check if it is properly transliterating.
"""


def build_user_content(batch: list[dict]) -> str:
    lines = ["Review the following rows:\n"]
    for row in batch:
        lines.append(
            f"message_id: {row['message_id']}\n"
            f"urdu_input: {row['user_message']}\n"
            f"roman_output: {row['romanized_text']}\n"
        )
    return "\n".join(lines)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_layer1_results(path: str) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for r in rows:
        r["flagged"] = str(r.get("flagged", "")).strip().lower() == "true"
    return rows


def select_rows_for_layer2(
    all_rows: list[dict], scope: str, sample_size: int, seed: int = 42
) -> list[dict]:
    if scope == "all":
        print(f"[select] Scope='all' — every row goes to Layer 2: {len(all_rows)}")
        return list(all_rows)

    flagged = [r for r in all_rows if r["flagged"]]
    unflagged = [r for r in all_rows if not r["flagged"]]

    rng = random.Random(seed)
    sample_size = min(sample_size, len(unflagged))
    sampled_unflagged = rng.sample(unflagged, sample_size)

    print(f"[select] Scope='triage' — Flagged: {len(flagged)}, Sampled unflagged: {len(sampled_unflagged)}")
    print(f"[select] Total rows going to OpenAI: {len(flagged) + len(sampled_unflagged)}")

    return flagged + sampled_unflagged


def load_checkpoint(checkpoint_path: Path) -> set[str]:
    if not checkpoint_path.exists():
        return set()
    with open(checkpoint_path, encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def append_checkpoint(checkpoint_path: Path, message_ids: list[str]) -> None:
    with open(checkpoint_path, "a", encoding="utf-8") as f:
        for mid in message_ids:
            f.write(f"{mid}\n")


# ── OpenAI call ────────────────────────────────────────────────────────────────

def call_openai_batch(
    client: "OpenAI",
    model: str,
    batch: list[dict],
    service_tier: str,
) -> list[RowVerdict]:
    """
    Sends one batch to OpenAI with structured output enforced via a
    Pydantic response_format. Validates that the returned verdict count
    AND message_id set exactly match the input batch — raises if not,
    so the caller can retry/split.
    """
    user_content = build_user_content(batch)

    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_format=BatchVerdicts,
        temperature=0.1,  # low temperature — judgment task, not creative
    )
    if service_tier != "auto":
        kwargs["service_tier"] = service_tier

    completion = client.chat.completions.parse(**kwargs)

    parsed = completion.choices[0].message.parsed
    if parsed is None:
        refusal = completion.choices[0].message.refusal
        raise ValueError(f"Model returned no parsed output. Refusal: {refusal}")

    verdicts = parsed.verdicts

    expected_ids = {int(r["message_id"]) for r in batch}
    got_ids = {v.message_id for v in verdicts}

    if expected_ids != got_ids:
        missing = expected_ids - got_ids
        extra = got_ids - expected_ids
        raise ValueError(
            f"Verdict message_id mismatch. Missing: {missing}. Unexpected extra: {extra}."
        )

    return verdicts


def process_batch_with_fallback(
    client: "OpenAI",
    model: str,
    batch: list[dict],
    service_tier: str,
    max_retries: int = 2,
) -> tuple[list[RowVerdict], float]:
    t0 = time.perf_counter()
    last_exc = None

    for attempt in range(max_retries + 1):
        try:
            verdicts = call_openai_batch(client, model, batch, service_tier)
            elapsed = time.perf_counter() - t0
            return verdicts, elapsed
        except Exception as exc:
            last_exc = exc
            print(
                f"[WARN] Batch of {len(batch)} failed on service_tier='{service_tier}' "
                f"(attempt {attempt + 1}/{max_retries + 1}): {exc}",
                file=sys.stderr,
            )

    if len(batch) == 1:
        # Last resort: if we were on flex and exhausted retries, try once
        # on standard tier before giving up — a capacity blip on flex
        # shouldn't cost us a row we could easily afford at standard price.
        if service_tier == "flex":
            print(
                f"[WARN] Row {batch[0]['message_id']} exhausted flex retries. "
                f"Trying once on standard tier before giving up.",
                file=sys.stderr,
            )
            try:
                verdicts = call_openai_batch(client, model, batch, "default")
                elapsed = time.perf_counter() - t0
                return verdicts, elapsed
            except Exception as exc:
                last_exc = exc
                print(f"[WARN] Standard-tier fallback also failed: {exc}", file=sys.stderr)

        print(
            f"[ERROR] Row {batch[0]['message_id']} failed after all retries: "
            f"{last_exc}. Marking as 'uncertain' with error note.",
            file=sys.stderr,
        )
        elapsed = time.perf_counter() - t0
        fallback = RowVerdict(
            message_id=int(batch[0]["message_id"]),
            verdict="uncertain",
            issue_type=None,
            reason=f"OpenAI call failed after retries: {last_exc}",
        )
        return [fallback], elapsed

    mid = len(batch) // 2
    print(f"[WARN] Splitting batch of {len(batch)} into {mid} + {len(batch) - mid} and retrying.")
    left, left_t = process_batch_with_fallback(client, model, batch[:mid], service_tier, max_retries)
    right, right_t = process_batch_with_fallback(client, model, batch[mid:], service_tier, max_retries)
    return left + right, left_t + right_t


def main():
    parser = argparse.ArgumentParser(description="Layer 2: OpenAI-based suspicious-pattern triage")
    parser.add_argument("--layer1-csv", required=True, help="Path to Layer 1 output CSV")
    parser.add_argument("--output", default="layer2_results.csv", help="Output CSV path")
    parser.add_argument("--scope", default="triage", choices=["all", "triage"],
                         help="'all' = every row; 'triage' = flagged + sampled unflagged only")
    parser.add_argument("--sample-size", type=int, default=100, help="Only used with --scope triage")
    parser.add_argument("--batch-size", type=int, default=150, help="Rows per OpenAI call")
    parser.add_argument("--model", default="gpt-5.4", help="OpenAI model name")
    parser.add_argument("--service-tier", default="flex", choices=["auto", "default", "flex", "priority"],
                         help="'flex' = ~50%% cheaper, slower, occasional capacity errors (recommended for non-urgent eval work). 'default' = standard pricing/speed.")
    parser.add_argument("--timeout", type=float, default=900.0,
                         help="Per-request timeout in seconds. Flex tier can be slow — OpenAI recommends up to 900s (15 min).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        print("[ERROR] Set OPENAI_API_KEY environment variable first.", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output)
    checkpoint_path = Path(str(output_path) + ".checkpoint")
    metrics_path = Path(str(output_path) + ".metrics.json")
    selection_record_path = Path(str(output_path) + ".selection_record.csv")

    all_rows = load_layer1_results(args.layer1_csv)
    selected = select_rows_for_layer2(all_rows, args.scope, args.sample_size, args.seed)

    with open(selection_record_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["message_id", "user_message", "romanized_text", "layer1_flags", "selection_reason"])
        for r in selected:
            if args.scope == "all":
                reason = "full_coverage"
            else:
                reason = "layer1_flagged" if r["flagged"] else "random_sample"
            w.writerow([r["message_id"], r["user_message"], r["romanized_text"], r.get("flags", ""), reason])
    print(f"[main] Selection record written to: {selection_record_path}")

    done_ids = load_checkpoint(checkpoint_path)
    if done_ids:
        print(f"[resume] Found checkpoint with {len(done_ids)} already-processed rows. Skipping those.")

    pending = [r for r in selected if r["message_id"] not in done_ids]
    print(f"[main] Rows remaining to process this run: {len(pending)}")

    if not pending:
        print("[main] Nothing to do — all selected rows already processed.")
        return

    client = OpenAI()  # reads OPENAI_API_KEY from env automatically

    write_header = not output_path.exists() or output_path.stat().st_size == 0
    out_f = open(output_path, "a", encoding="utf-8", newline="")
    writer = csv.writer(out_f)
    if write_header:
        writer.writerow([
            "message_id", "user_message", "romanized_text", "layer1_flags",
            "openai_verdict", "openai_issue_type", "openai_reason",
        ])
        out_f.flush()

    batch_latencies = []
    verdict_counts: dict[str, int] = {}
    issue_type_counts: dict[str, int] = {}
    row_count_processed = 0
    run_start = time.perf_counter()

    n_batches = (len(pending) + args.batch_size - 1) // args.batch_size

    for batch_idx in range(n_batches):
        start = batch_idx * args.batch_size
        end = start + args.batch_size
        batch = pending[start:end]

        print(f"[batch {batch_idx + 1}/{n_batches}] Sending {len(batch)} rows to {args.model} ...")
        verdicts, elapsed = process_batch_with_fallback(client, args.model, batch, args.service_tier)
        batch_latencies.append(elapsed)
        print(f"[batch {batch_idx + 1}/{n_batches}] Done in {elapsed:.1f}s")

        verdict_by_id = {v.message_id: v for v in verdicts}
        completed_ids = []
        for row in batch:
            mid = int(row["message_id"])
            v = verdict_by_id.get(mid)
            if v is None:
                print(f"[WARN] No verdict returned for message_id {mid}, marking uncertain.", file=sys.stderr)
                v = RowVerdict(message_id=mid, verdict="uncertain", issue_type=None, reason="No verdict returned.")

            verdict_counts[v.verdict] = verdict_counts.get(v.verdict, 0) + 1
            if v.issue_type:
                issue_type_counts[v.issue_type] = issue_type_counts.get(v.issue_type, 0) + 1

            writer.writerow([
                row["message_id"], row["user_message"], row["romanized_text"],
                row.get("flags", ""), v.verdict, v.issue_type or "", v.reason,
            ])
            completed_ids.append(row["message_id"])
            row_count_processed += 1

        out_f.flush()
        append_checkpoint(checkpoint_path, completed_ids)

    out_f.close()
    total_time = time.perf_counter() - run_start

    metrics = {
        "scope": args.scope,
        "model": args.model,
        "rows_processed_this_run": row_count_processed,
        "total_rows_in_output": len(done_ids) + row_count_processed,
        "total_wall_clock_seconds": round(total_time, 2),
        "num_batches": len(batch_latencies),
        "avg_latency_per_batch_seconds": round(sum(batch_latencies) / len(batch_latencies), 2) if batch_latencies else None,
        "min_batch_latency_seconds": round(min(batch_latencies), 2) if batch_latencies else None,
        "max_batch_latency_seconds": round(max(batch_latencies), 2) if batch_latencies else None,
        "verdict_counts": verdict_counts,
        "issue_type_counts": issue_type_counts,
    }
    with open(metrics_path, "w", encoding="utf-8") as mf:
        json.dump(metrics, mf, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("LAYER 2 COMPLETE")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"\nResults written to: {output_path}")
    print(f"Metrics written to: {metrics_path}")
    print(f"\nNext: review rows where openai_verdict is 'suspicious' or 'uncertain' — this is Layer 3.")


if __name__ == "__main__":
    main()