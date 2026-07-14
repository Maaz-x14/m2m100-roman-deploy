"""
eval_layer1.py
--------------
Layer 1 of the evaluation pipeline: sends all user_message rows through the
locally running /romanize endpoint in batches, applies automated heuristic
flags, and records latency metrics for benchmarking.

USAGE
-----
python eval_layer1.py \
    --csv ../../data/raw/faiq-data.csv \
    --url http://localhost:2000/romanize \
    --batch-size 150 \
    --output ../../data/layer1/layer1_results.csv

RESUMABLE
---------
Writes results incrementally after every successful batch, and maintains a
checkpoint file (<output>.checkpoint) listing message_ids already processed.
If the script crashes or is interrupted (Ctrl+C), re-running the same command
skips completed rows and continues from where it left off — no repeated API
calls, no lost work.

OUTPUT COLUMNS
--------------
message_id, user_message, romanized_text, flags, flagged

flags is a semicolon-separated string (e.g. "script_leakage;length_outlier")
flagged is True/False — True if flags is non-empty

METRICS
-------
Printed to stdout at the end, and written to <output>.metrics.json:
  - total rows processed
  - total wall-clock time
  - avg latency per row (batch_latency / batch_size)
  - avg latency per batch
  - min / max batch latency
  - throughput (rows/sec)
  - flag counts by type
"""

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import requests

# ── Heuristic flag thresholds ─────────────────────────────────────────────────
# Roman Urdu output is typically ~1.3-1.8x the character count of Urdu input.
# Widened band to reduce false positives on short messages (short strings have
# high variance in this ratio just from rounding).
LENGTH_RATIO_MIN = 0.8
LENGTH_RATIO_MAX = 2.5
MIN_LEN_FOR_RATIO_CHECK = 8  # skip ratio check on very short inputs — too noisy

# Arabic/Urdu Unicode block — covers Arabic, Arabic Supplement, Arabic
# Presentation Forms (the ranges Urdu script actually uses)
ARABIC_SCRIPT_RE = re.compile(
    r'[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]'
)

# Degenerate output: same 3+ word chunk repeated back-to-back 3+ times
REPEAT_LOOP_RE = re.compile(r'\b(\w+(?:\s+\w+){0,3})\b(?:\s+\1\b){2,}', re.IGNORECASE)


def _reconstruct_rows(csv_path: str, malformed_log_path: Path) -> tuple[list[str], list[dict]]:
    """
    Reconstructs CSV rows WITHOUT relying on quote-aware parsing.

    WHY: pandas / csv.DictReader both use quote-balance to detect field and
    row boundaries. This source file has at least one stray/unescaped '"'
    character inside free-text Urdu content, which desyncs quote-aware
    parsers for the rest of the file — silently collapsing thousands of
    rows into a few hundred malformed ones. That is unacceptable data loss.

    APPROACH: a new logical row is detected when a physical line BEGINS
    with "<digits><delimiter>" — i.e. looks like the start of a fresh
    message_id. All physical lines between one such boundary and the next
    are joined (preserving embedded newlines) into a single buffer, which
    is then split into fields.

    Field-count alone (an earlier version of this function) is NOT a
    reliable row-boundary signal: a multi-line field's FIRST physical line
    can already contain enough delimiters to match the expected column
    count on its own, causing the row to be flushed one line too early and
    swallowing the next row's message_id into the tail of the previous
    field. Anchoring on the message_id prefix avoids this entirely.

    Any reconciled row whose final field count doesn't match the header's
    column count is written to <output>.malformed_rows.txt for manual
    review — NOT silently dropped.

    Returns (header_columns, list_of_row_dicts_from_reconciled_rows).
    """
    with open(csv_path, encoding="utf-8-sig") as f:
        raw_lines = f.read().split("\n")

    if not raw_lines:
        print("[ERROR] CSV file is empty.", file=sys.stderr)
        sys.exit(1)

    delimiter = ";" if raw_lines[0].count(";") >= raw_lines[0].count(",") else ","
    print(f"[load_rows] Detected delimiter: '{delimiter}'")

    header = raw_lines[0].split(delimiter)
    expected_cols = len(header)
    print(f"[load_rows] Detected {expected_cols} columns: {header}")

    row_start_re = re.compile(r"^\d+" + re.escape(delimiter))

    rows: list[dict] = []
    malformed: list[str] = []
    buffer_lines: list[str] = []

    def _flush(buf_lines: list[str]) -> None:
        if not buf_lines:
            return
        buffer = "\n".join(buf_lines)
        fields = buffer.split(delimiter)
        if len(fields) == expected_cols:
            rows.append(dict(zip(header, fields)))
        else:
            malformed.append(
                f"[expected {expected_cols} fields, got {len(fields)}] {buffer}"
            )

    for line in raw_lines[1:]:
        if row_start_re.match(line):
            # New row boundary found — flush whatever was buffered before it
            _flush(buffer_lines)
            buffer_lines = [line]
        else:
            if not buffer_lines:
                # Junk before any recognizable row start (e.g. leading blank
                # lines) — skip rather than let it corrupt the first real row
                if line.strip():
                    malformed.append(f"[no preceding row start] {line}")
                continue
            buffer_lines.append(line)

    _flush(buffer_lines)  # flush the final row at EOF

    if malformed:
        with open(malformed_log_path, "w", encoding="utf-8") as mf:
            mf.write(f"# {len(malformed)} row(s) could not be reconciled to {expected_cols} columns.\n")
            mf.write("# Reviewed manually — these were NOT included in Layer 1 processing.\n\n")
            for m in malformed:
                mf.write(m + "\n\n")
        print(
            f"[load_rows] WARNING: {len(malformed)} row(s) could not be reconciled "
            f"and were written to '{malformed_log_path}' for manual review — "
            f"NOT silently dropped."
        )

    print(f"[load_rows] Successfully reconstructed {len(rows)} row(s) out of "
          f"{len(rows) + len(malformed)} attempted.")

    return header, rows


def load_rows(csv_path: str, malformed_log_path: Path) -> list[dict]:
    """
    Reads the CSV via quote-immune reconstruction (see _reconstruct_rows),
    keeps only message_id + user_message, dedupes, drops null/blank rows.
    Prints counts of everything dropped — nothing disappears silently.
    """
    header, raw_rows = _reconstruct_rows(csv_path, malformed_log_path)

    if "message_id" not in header or "user_message" not in header:
        print(
            f"[ERROR] CSV must contain 'message_id' and 'user_message' columns. "
            f"Found: {header}",
            file=sys.stderr,
        )
        sys.exit(1)

    total = len(raw_rows)

    # ── Validate message_id is a clean integer ──────────────────────────────
    def _is_clean_int(x: str) -> bool:
        return bool(re.fullmatch(r"\d+", str(x).strip()))

    bad_id_rows = [r for r in raw_rows if not _is_clean_int(r.get("message_id", ""))]
    dropped_bad_id = len(bad_id_rows)
    if dropped_bad_id:
        bad_id_log = str(malformed_log_path).replace(".txt", "") + ".bad_message_id.txt"
        with open(bad_id_log, "w", encoding="utf-8") as bf:
            bf.write(f"# {dropped_bad_id} row(s) had a non-numeric message_id after reconstruction.\n\n")
            for r in bad_id_rows:
                bf.write(f"{r}\n\n")
        print(
            f"[load_rows] WARNING: {dropped_bad_id} row(s) had non-numeric "
            f"message_id even after reconstruction — logged to '{bad_id_log}'."
        )

    clean_rows = [r for r in raw_rows if _is_clean_int(r.get("message_id", ""))]

    # ── Drop blank / null user_message ──────────────────────────────────────
    rows = [
        r for r in clean_rows
        if r.get("user_message") is not None and r["user_message"].strip() != ""
    ]
    dropped_blank = len(clean_rows) - len(rows)

    # ── Dedupe by exact user_message text, keep lowest message_id ───────────
    rows.sort(key=lambda r: int(r["message_id"]))
    seen = set()
    deduped = []
    for r in rows:
        text = r["user_message"].strip()
        if text in seen:
            continue
        seen.add(text)
        deduped.append({"message_id": r["message_id"], "user_message": text})
    dropped_dupes = len(rows) - len(deduped)

    print(f"[load_rows] Total rows reconstructed    : {total}")
    print(f"[load_rows] Dropped (bad message_id)     : {dropped_bad_id}")
    print(f"[load_rows] Dropped (blank/null)         : {dropped_blank}")
    print(f"[load_rows] Dropped (duplicate text)     : {dropped_dupes}")
    print(f"[load_rows] Remaining rows to process    : {len(deduped)}")

    return deduped


def load_checkpoint(checkpoint_path: Path) -> set[str]:
    if not checkpoint_path.exists():
        return set()
    with open(checkpoint_path, encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def append_checkpoint(checkpoint_path: Path, message_ids: list[str]) -> None:
    with open(checkpoint_path, "a", encoding="utf-8") as f:
        for mid in message_ids:
            f.write(f"{mid}\n")


def compute_flags(urdu: str, roman: str) -> list[str]:
    """
    Applies Layer 1 heuristic checks. Returns a list of flag names.
    Empty list = looks fine (still subject to Layer 2 / Layer 3 review).
    """
    flags = []

    # 1. Script leakage — output still contains Urdu/Arabic script characters
    if ARABIC_SCRIPT_RE.search(roman):
        flags.append("script_leakage")

    # 2. Degenerate output — empty, or a repeated loop pattern
    if not roman.strip():
        flags.append("empty_output")
    elif REPEAT_LOOP_RE.search(roman):
        flags.append("repeat_loop")

    # 3. Length-ratio outlier — skip check on very short inputs (noisy)
    urdu_len = len(urdu.strip())
    roman_len = len(roman.strip())
    if urdu_len >= MIN_LEN_FOR_RATIO_CHECK and roman_len > 0:
        ratio = roman_len / urdu_len
        if ratio < LENGTH_RATIO_MIN or ratio > LENGTH_RATIO_MAX:
            flags.append("length_ratio_outlier")
    elif roman_len == 0 and urdu_len > 0:
        # already caught by empty_output above, avoid double-counting silently
        pass

    return flags


def call_romanize_batch(url: str, texts: list[str], timeout: int) -> list[str]:
    """
    POSTs a batch of texts to /romanize. Returns the list of romanized
    strings in the same order. Raises on HTTP or shape errors — caller
    handles retry/fallback.
    """
    resp = requests.post(url, json={"text": texts}, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    result = data.get("romanized_text")
    if not isinstance(result, list):
        raise ValueError(f"Expected list response for batch input, got: {type(result)}")
    if len(result) != len(texts):
        raise ValueError(
            f"Row count mismatch: sent {len(texts)}, got back {len(result)}"
        )
    return result


def process_batch_with_fallback(
    url: str, batch: list[dict], timeout: int
) -> tuple[list[str], float]:
    """
    Attempts the full batch in one call. If it fails (timeout, OOM, error),
    falls back to splitting the batch in half and retrying each half
    recursively — down to single-row calls if necessary. This means one bad
    row (e.g. unexpectedly long) doesn't lose the whole batch's progress.

    Returns (romanized_texts_in_order, elapsed_seconds).
    """
    texts = [r["user_message"] for r in batch]
    t0 = time.perf_counter()
    try:
        results = call_romanize_batch(url, texts, timeout)
        elapsed = time.perf_counter() - t0
        return results, elapsed
    except Exception as exc:
        if len(batch) == 1:
            # Single row still failing — record the error inline rather than
            # crashing the whole run. Downstream flag will catch empty output.
            print(f"[WARN] Row {batch[0]['message_id']} failed: {exc}", file=sys.stderr)
            elapsed = time.perf_counter() - t0
            return [""], elapsed

        mid = len(batch) // 2
        print(
            f"[WARN] Batch of {len(batch)} failed ({exc}); splitting into "
            f"{mid} + {len(batch) - mid} and retrying.",
            file=sys.stderr,
        )
        left_results, left_time = process_batch_with_fallback(url, batch[:mid], timeout)
        right_results, right_time = process_batch_with_fallback(url, batch[mid:], timeout)
        return left_results + right_results, left_time + right_time


def main():
    parser = argparse.ArgumentParser(description="Layer 1 evaluation: batch romanization + heuristic flags")
    parser.add_argument("--csv", required=True, help="Path to faiq-data.csv")
    parser.add_argument("--url", default="http://localhost:2000/romanize", help="Romanize API endpoint")
    parser.add_argument("--batch-size", type=int, default=16, help="Rows per HTTP call")
    parser.add_argument("--output", default="../../data/layer1/layer1_results.csv", help="Output CSV path")
    parser.add_argument("--timeout", type=int, default=1800, help="Per-batch HTTP timeout in seconds (CPU can be slow)")
    args = parser.parse_args()

    output_path = Path(args.output)
    checkpoint_path = Path(str(output_path) + ".checkpoint")
    metrics_path = Path(str(output_path) + ".metrics.json")

    malformed_log_path = Path(str(output_path) + ".malformed_rows.txt")
    rows = load_rows(args.csv, malformed_log_path)
    done_ids = load_checkpoint(checkpoint_path)
    if done_ids:
        print(f"[resume] Found checkpoint with {len(done_ids)} already-processed rows. Skipping those.")

    pending = [r for r in rows if r["message_id"] not in done_ids]
    print(f"[main] Rows remaining to process this run: {len(pending)}")

    if not pending:
        print("[main] Nothing to do — all rows already processed.")
        return

    # Open output CSV in append mode; write header only if file is new
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    out_f = open(output_path, "a", encoding="utf-8", newline="")
    writer = csv.writer(out_f)
    if write_header:
        writer.writerow(["message_id", "user_message", "romanized_text", "flags", "flagged"])
        out_f.flush()

    # ── Metrics accumulators ──────────────────────────────────────────────────
    batch_latencies = []
    row_count_processed = 0
    flagged_row_count = 0
    flag_counts: dict[str, int] = {}
    run_start = time.perf_counter()

    n_batches = (len(pending) + args.batch_size - 1) // args.batch_size

    for batch_idx in range(n_batches):
        start = batch_idx * args.batch_size
        end = start + args.batch_size
        batch = pending[start:end]

        print(f"[batch {batch_idx + 1}/{n_batches}] Sending {len(batch)} rows ...")
        results, elapsed = process_batch_with_fallback(args.url, batch, args.timeout)
        batch_latencies.append(elapsed)
        print(f"[batch {batch_idx + 1}/{n_batches}] Done in {elapsed:.1f}s "
              f"({elapsed / len(batch):.2f}s/row avg)")

        completed_ids = []
        for row, roman in zip(batch, results):
            flags = compute_flags(row["user_message"], roman)
            for f in flags:
                flag_counts[f] = flag_counts.get(f, 0) + 1
            if flags:
                flagged_row_count += 1
            writer.writerow([
                row["message_id"],
                row["user_message"],
                roman,
                ";".join(flags),
                bool(flags),
            ])
            completed_ids.append(row["message_id"])
            row_count_processed += 1

        out_f.flush()
        append_checkpoint(checkpoint_path, completed_ids)

    out_f.close()

    total_time = time.perf_counter() - run_start

    # ── Metrics summary ───────────────────────────────────────────────────────
    metrics = {
        "rows_processed_this_run": row_count_processed,
        "total_rows_in_output": len(done_ids) + row_count_processed,
        "total_wall_clock_seconds": round(total_time, 2),
        "num_batches": len(batch_latencies),
        "avg_latency_per_row_seconds": round(total_time / row_count_processed, 3) if row_count_processed else None,
        "avg_latency_per_batch_seconds": round(sum(batch_latencies) / len(batch_latencies), 2) if batch_latencies else None,
        "min_batch_latency_seconds": round(min(batch_latencies), 2) if batch_latencies else None,
        "max_batch_latency_seconds": round(max(batch_latencies), 2) if batch_latencies else None,
        "throughput_rows_per_second": round(row_count_processed / total_time, 3) if total_time > 0 else None,
        "flag_counts": flag_counts,
        "total_flagged_rows": flagged_row_count,
    }

    with open(metrics_path, "w", encoding="utf-8") as mf:
        json.dump(metrics, mf, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("LAYER 1 COMPLETE")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"\nResults written to: {output_path}")
    print(f"Metrics written to: {metrics_path}")


if __name__ == "__main__":
    main()