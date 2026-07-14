"""
filter_issue_type.py
---------------------
Filters layer2_results.csv down to rows matching a specific
openai_issue_type (e.g. phonetic_mismatch) for focused review /
fine-tuning data generation.

USAGE
-----
python filter_issue_type.py \
    --input ../../data/layer2/layer2_results.csv \
    --issue-type phonetic_mismatch \
    --output ../../data/wordlists/phonetic_mismatch_rows.csv

Defaults to --issue-type phonetic_mismatch and --output
../../data/wordlists/<issue_type>_rows.csv if not specified.
"""

import argparse
import csv
import sys


def main():
    parser = argparse.ArgumentParser(description="Filter Layer 2 results by openai_issue_type")
    parser.add_argument("--input", default="../../data/layer2/layer2_results.csv", help="Path to layer2_results.csv")
    parser.add_argument("--issue-type", default="phonetic_mismatch",
                         help="Value of openai_issue_type to filter for")
    parser.add_argument("--output", default=None,
                         help="Output CSV path (default: ../../data/wordlists/<issue_type>_rows.csv)")
    args = parser.parse_args()

    output_path = args.output or f"../../data/wordlists/{args.issue_type}_rows.csv"

    with open(args.input, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if fieldnames is None or "openai_issue_type" not in fieldnames:
            print("[ERROR] Input CSV missing 'openai_issue_type' column.", file=sys.stderr)
            sys.exit(1)
        rows = [r for r in reader if r.get("openai_issue_type", "").strip() == args.issue_type]

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[filter_issue_type] Matched {len(rows)} rows with openai_issue_type='{args.issue_type}'")
    print(f"[filter_issue_type] Written to: {output_path}")


if __name__ == "__main__":
    main()
