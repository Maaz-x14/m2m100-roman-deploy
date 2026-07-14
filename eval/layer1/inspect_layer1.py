"""
inspect_layer1.py
------------------
Summarizes layer1_results.csv so you have real numbers before estimating
Gemini/OpenAI cost or deciding full-coverage vs sampled Layer 2 scope.

USAGE
-----
python inspect_layer1.py --csv ../../data/layer1/layer1_results.csv
"""

import argparse
import csv
import statistics
import sys


def main():
    parser = argparse.ArgumentParser(description="Inspect Layer 1 results before cost estimation")
    parser.add_argument("--csv", required=True, help="Path to layer1_results.csv")
    parser.add_argument("--show-samples", type=int, default=3, help="Number of flagged rows to print")
    args = parser.parse_args()

    with open(args.csv, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("[ERROR] No rows found in file.", file=sys.stderr)
        sys.exit(1)

    total = len(rows)
    flagged_rows = [r for r in rows if str(r.get("flagged", "")).strip().lower() == "true"]
    unflagged_rows = [r for r in rows if r not in flagged_rows]

    # Flag-type breakdown
    flag_type_counts: dict[str, int] = {}
    for r in flagged_rows:
        for f in str(r.get("flags", "")).split(";"):
            f = f.strip()
            if f:
                flag_type_counts[f] = flag_type_counts.get(f, 0) + 1

    # Length stats (character counts — proxy for token counts)
    urdu_lens = [len(r.get("user_message", "")) for r in rows]
    roman_lens = [len(r.get("romanized_text", "")) for r in rows]

    print("=" * 60)
    print("LAYER 1 RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total rows                    : {total}")
    print(f"Flagged rows                  : {len(flagged_rows)} ({100*len(flagged_rows)/total:.1f}%)")
    print(f"Unflagged rows                : {len(unflagged_rows)} ({100*len(unflagged_rows)/total:.1f}%)")
    print()
    print("Flag type breakdown:")
    for k, v in sorted(flag_type_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k}: {v}")
    print()
    print("Urdu input length (characters):")
    print(f"  min={min(urdu_lens)}  max={max(urdu_lens)}  "
          f"mean={statistics.mean(urdu_lens):.1f}  median={statistics.median(urdu_lens):.1f}")
    print("Roman output length (characters):")
    print(f"  min={min(roman_lens)}  max={max(roman_lens)}  "
          f"mean={statistics.mean(roman_lens):.1f}  median={statistics.median(roman_lens):.1f}")
    print()

    if args.show_samples and flagged_rows:
        print(f"Sample flagged rows (showing {min(args.show_samples, len(flagged_rows))}):")
        for r in flagged_rows[:args.show_samples]:
            print(f"  [{r['message_id']}] flags={r.get('flags','')}")
            print(f"    urdu : {r.get('user_message','')[:80]}")
            print(f"    roman: {r.get('romanized_text','')[:80]}")
        print()

    print("=" * 60)
    print("Use these numbers with estimate_cost.py to size your Layer 2 budget.")


if __name__ == "__main__":
    main()
