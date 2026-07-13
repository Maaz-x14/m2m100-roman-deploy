#!/usr/bin/env python3
"""
Remove repeated words/lines from incorrect_words_claude.md.

Each line in the file is treated as one entry (a single Urdu word or short
phrase). This script keeps only the FIRST occurrence of each unique line,
preserving original order, and strips blank lines / surrounding whitespace.

Usage:
    python remove_reps.py <input_path> [output_path]

If output_path is omitted, writes to <input_stem>_deduped.md next to the input.
"""

import sys
from pathlib import Path


def dedupe_file(input_path: Path, output_path: Path) -> tuple[int, int]:
    seen = set()
    unique_lines = []

    with input_path.open("r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    total = 0
    for line in raw_lines:
        word = line.strip()
        if not word:
            continue
        total += 1
        if word not in seen:
            seen.add(word)
            unique_lines.append(word)

    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(unique_lines) + "\n")

    return total, len(unique_lines)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        sys.exit(1)

    if len(sys.argv) >= 3:
        output_path = Path(sys.argv[2])
    else:
        output_path = input_path.with_name(f"{input_path.stem}_deduped{input_path.suffix}")

    total, unique = dedupe_file(input_path, output_path)
    removed = total - unique

    print(f"Total non-empty lines : {total}")
    print(f"Unique lines kept     : {unique}")
    print(f"Duplicates removed    : {removed}")
    print(f"Output written to     : {output_path}")


if __name__ == "__main__":
    main()
