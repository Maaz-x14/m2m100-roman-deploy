"""
estimate_cost.py
-----------------
Estimates OpenAI API cost for running Layer 2 triage over layer1_results.csv,
BEFORE you spend anything. Uses tiktoken for accurate token counts when
available, with a clearly-labeled character-based fallback otherwise.

PRICING DISCLAIMER
-------------------
The prices below are current as of July 2026, sourced from OpenAI's public
pricing page and third-party trackers. Prices change. VERIFY against
https://openai.com/api/pricing/ before committing to a run, especially if
you're reading this weeks/months after it was written.

USAGE
-----
python estimate_cost.py \
    --csv ../../data/layer1/layer1_results.csv \
    --model gpt-5.4-mini \
    --scope all              # or: triage (flagged + sample only)
    --sample-size 100        # only used if --scope triage
    --batch-size 150
"""

import argparse
import csv
import sys

# ── Pricing table: USD per 1M tokens (input, output) ────────────────────────
# Source: OpenAI pricing page + cross-checked trackers, July 2026.
# VERIFY at https://openai.com/api/pricing/ before spending real money.
PRICING = {
    "gpt-5.6-luna":   (1.00, 6.00),
    "gpt-5.4-nano":   (0.20, 1.25),
    "gpt-5.4-mini":   (0.75, 4.50),
    "gpt-5.4":        (2.50, 15.00),
    "gpt-5.5":        (5.00, 30.00),
    "gpt-4.1-nano":   (0.10, 0.40),
    "gpt-4.1-mini":   (0.40, 1.60),
    "gpt-4.1":        (2.00, 8.00),
}

# System prompt is roughly constant regardless of dataset — measured once
# against the actual Layer 2 SYSTEM_PROMPT text (see eval_layer2_openai.py).
# This is an approximation; the estimator recomputes it exactly if tiktoken
# is available by importing the real prompt string.
FALLBACK_CHARS_PER_TOKEN = 4.0  # rough English/mixed-text approximation


def get_token_counter(model_hint: str):
    """
    Returns a function(text) -> int token count.
    Tries tiktoken first (accurate); falls back to a character-based
    approximation if tiktoken can't load its vocab file (e.g. no internet
    access to its blob storage, or not installed).
    """
    try:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model(model_hint)
        except KeyError:
            enc = tiktoken.get_encoding("o200k_base")  # current default for gpt-4o/5.x family
        def _count(text: str) -> int:
            return len(enc.encode(text))
        print("[estimate_cost] Using tiktoken for accurate token counts.")
        return _count, True
    except Exception as exc:
        print(
            f"[estimate_cost] WARNING: tiktoken unavailable ({exc}). "
            f"Falling back to character-based approximation "
            f"(~{FALLBACK_CHARS_PER_TOKEN} chars/token). This is a rough "
            f"estimate only — install/verify tiktoken for accuracy.",
            file=sys.stderr,
        )
        def _count(text: str) -> int:
            return int(len(text) / FALLBACK_CHARS_PER_TOKEN)
        return _count, False


def load_rows(csv_path: str) -> list[dict]:
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for r in rows:
        r["flagged"] = str(r.get("flagged", "")).strip().lower() == "true"
    return rows


def main():
    parser = argparse.ArgumentParser(description="Estimate OpenAI API cost for Layer 2")
    parser.add_argument("--csv", required=True, help="Path to layer1_results.csv")
    parser.add_argument("--model", default="gpt-5.4-mini", choices=list(PRICING.keys()))
    parser.add_argument("--scope", default="all", choices=["all", "triage"],
                         help="'all' = every row; 'triage' = flagged + sampled unflagged only")
    parser.add_argument("--sample-size", type=int, default=100, help="Only used with --scope triage")
    parser.add_argument("--batch-size", type=int, default=150)
    parser.add_argument("--avg-output-tokens-per-row", type=int, default=45,
                         help="Estimated output tokens per verdict (message_id + verdict + issue_type + short reason)")
    args = parser.parse_args()

    if args.model not in PRICING:
        print(f"[ERROR] Unknown model '{args.model}'. Choose from: {list(PRICING.keys())}", file=sys.stderr)
        sys.exit(1)

    input_price, output_price = PRICING[args.model]
    count_tokens, is_accurate = get_token_counter(args.model)

    rows = load_rows(args.csv)
    flagged = [r for r in rows if r["flagged"]]
    unflagged = [r for r in rows if not r["flagged"]]

    if args.scope == "all":
        target_rows = rows
    else:
        target_rows = flagged + unflagged[:min(args.sample_size, len(unflagged))]

    n_rows = len(target_rows)
    n_batches = (n_rows + args.batch_size - 1) // args.batch_size

    # ── System prompt token count ────────────────────────────────────────────
    # Uses the REAL system prompt text from eval_layer2_openai.py for an
    # exact count, rather than guessing its length.
    try:
        from eval_layer2_openai import SYSTEM_PROMPT
        system_prompt_tokens = count_tokens(SYSTEM_PROMPT)
    except ImportError:
        print(
            "[WARN] Could not import SYSTEM_PROMPT from eval_layer2_openai.py "
            "(must be in the same directory). Using a rough estimate instead.",
            file=sys.stderr,
        )
        system_prompt_tokens = int(3800 / FALLBACK_CHARS_PER_TOKEN)

    # ── Per-row input tokens (Urdu + Roman text) ─────────────────────────────
    total_row_input_tokens = 0
    for r in target_rows:
        row_text = (
            f"message_id: {r['message_id']}\n"
            f"urdu_input: {r.get('user_message','')}\n"
            f"roman_output: {r.get('romanized_text','')}\n"
        )
        total_row_input_tokens += count_tokens(row_text)

    total_input_tokens = total_row_input_tokens + (system_prompt_tokens * n_batches)
    total_output_tokens = n_rows * args.avg_output_tokens_per_row

    input_cost = (total_input_tokens / 1_000_000) * input_price
    output_cost = (total_output_tokens / 1_000_000) * output_price
    total_cost = input_cost + output_cost

    print("\n" + "=" * 60)
    print(f"COST ESTIMATE — model={args.model}  scope={args.scope}")
    print("=" * 60)
    print(f"Token counting method       : {'tiktoken (accurate)' if is_accurate else 'char-based approximation (rough)'}")
    print(f"Rows in scope                : {n_rows}")
    print(f"Batches (batch_size={args.batch_size})    : {n_batches}")
    print(f"System prompt tokens/batch    : ~{system_prompt_tokens} (approximate)")
    print(f"Total input tokens (est.)     : {total_input_tokens:,}")
    print(f"Total output tokens (est.)    : {total_output_tokens:,}  (assumes ~{args.avg_output_tokens_per_row} tokens/verdict)")
    print(f"Pricing                       : ${input_price}/M input, ${output_price}/M output")
    print(f"Estimated input cost          : ${input_cost:.4f}")
    print(f"Estimated output cost         : ${output_cost:.4f}")
    print(f"ESTIMATED TOTAL COST          : ${total_cost:.4f}")
    print("=" * 60)
    print("\nThis is an estimate, not a guarantee. Actual cost may be LOWER:")
    print("OpenAI applies automatic prompt caching (~90% discount) to repeated")
    print("prefixes like this system prompt when calls happen close together in")
    print("time — this estimate assumes NO caching (worst case). Actual cost")
    print("may also be HIGHER if batches get split/retried due to errors.")
    print("Verify current pricing at https://openai.com/api/pricing/ before running.")


if __name__ == "__main__":
    main()
