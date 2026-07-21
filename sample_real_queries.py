"""
sample_real_queries.py
------------------------
Replaces synthetic test sentences with REAL user queries sampled from
data/raw/faiq-data.csv (user_message column), instead of hand-written
Urdu sentences whose length/complexity was a guess. This directly
addresses the open question from benchmarking: synthetic short vs long
sentences gave different "best batch config" answers, and we didn't
know which matched real traffic. Real data resolves that instead of
guessing further.

FILTER: discards any query with fewer than MIN_WORDS words (default 5),
per instruction — short fragments like "ہوں" or "منڈی کا ریٹ." aren't
representative of a full voice-turn utterance and would skew latency
low in a way that doesn't reflect real usage.

Word count uses simple whitespace splitting. This is a reasonable proxy
for Urdu word count but not a linguistically precise tokenizer-based
count — flagging this as an approximation, not a verified exact measure.

USAGE:
    python sample_real_queries.py [--n N] [--min-words W] [--seed S]

    --n           how many sentences to sample (default 20)
    --min-words   minimum word count to keep a query (default 5)
    --seed        random seed for reproducibility (default 42)

OUTPUT:
    Prints the sampled sentences as a Python list literal, ready to
    paste directly into bench_phase2.py's SENTENCES variable.
    Also writes them to sampled_sentences.txt (one per line) and
    sampled_sentences.py (importable list) for convenience.
"""

import argparse
import csv
import random
import sys
from pathlib import Path

DEFAULT_CSV_PATH = "faiq-data.csv"


def load_queries(csv_path: str, min_words: int) -> list[dict]:
    """
    Reads faiq-data.csv, returns rows where user_message has >= min_words
    words (whitespace-split). Keeps message_id alongside for traceability.
    """
    kept = []
    total = 0
    too_short = 0
    empty = 0

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "user_message" not in reader.fieldnames:
            print(f"ERROR: 'user_message' column not found. Columns present: {reader.fieldnames}")
            sys.exit(1)

        for row in reader:
            total += 1
            text = (row.get("user_message") or "").strip()
            if not text:
                empty += 1
                continue
            word_count = len(text.split())
            if word_count < min_words:
                too_short += 1
                continue
            kept.append({
                "message_id": row.get("message_id", "?"),
                "user_message": text,
                "word_count": word_count,
            })

    print(f"Total rows read:        {total}")
    print(f"Empty user_message:     {empty}")
    print(f"Below {min_words} words:        {too_short}")
    print(f"Kept (>= {min_words} words):    {len(kept)}")
    return kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", default=DEFAULT_CSV_PATH)
    parser.add_argument("--n", type=int, default=20, help="How many sentences to sample")
    parser.add_argument("--min-words", type=int, default=5, help="Minimum word count to keep a query")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run this from the repo root, "
              f"or pass --csv-path to point at the file.")
        sys.exit(1)

    print("=" * 70)
    print(f"Sampling real user queries from {csv_path}")
    print("=" * 70)

    candidates = load_queries(str(csv_path), args.min_words)

    if len(candidates) < args.n:
        print(f"\nWARNING: only {len(candidates)} queries meet the {args.min_words}-word "
              f"minimum, but {args.n} were requested. Sampling all {len(candidates)} instead.")
        sample_size = len(candidates)
    else:
        sample_size = args.n

    random.seed(args.seed)
    sampled = random.sample(candidates, sample_size)

    word_counts = [s["word_count"] for s in sampled]
    print(f"\nSampled {len(sampled)} queries.")
    print(f"Word count range: {min(word_counts)}-{max(word_counts)} "
          f"(mean: {sum(word_counts)/len(word_counts):.1f})")

    print("\n" + "-" * 70)
    print("Sampled sentences (message_id: word_count: text)")
    print("-" * 70)
    for s in sampled:
        preview = s["user_message"][:80] + ("..." if len(s["user_message"]) > 80 else "")
        print(f"  [{s['message_id']}] ({s['word_count']}w): {preview}")

    # ── Write outputs ────────────────────────────────────────────────────
    txt_path = Path("sampled_sentences.txt")
    with txt_path.open("w", encoding="utf-8") as f:
        for s in sampled:
            f.write(s["user_message"] + "\n")
    print(f"\nWritten: {txt_path.resolve()}")

    py_path = Path("sampled_sentences.py")
    with py_path.open("w", encoding="utf-8") as f:
        f.write('"""Auto-generated by sample_real_queries.py — real user queries, do not hand-edit."""\n\n')
        f.write("SENTENCES = [\n")
        for s in sampled:
            escaped = s["user_message"].replace('"', '\\"')
            f.write(f'    "{escaped}",  # message_id={s["message_id"]}, {s["word_count"]} words\n')
        f.write("]\n")
    print(f"Written: {py_path.resolve()}")

    print("\n" + "=" * 70)
    print("Next step: import SENTENCES from sampled_sentences.py into")
    print("bench_phase2.py (replace the hand-written SENTENCES list), then")
    print("rerun run_sweep.py to test against realistic real-user query length.")
    print("=" * 70)


if __name__ == "__main__":
    main()
