#!/usr/bin/env python3
"""
run_benchmark.py
----------------
Generates Roman Urdu predictions for benchmark_dataset.csv using a given
model (either your fine-tuned LoRA adapter or a base HF model like Mavkif's).

Usage:
    # Your fine-tuned model (LoRA adapter dir)
    python run_benchmark.py \
        --model_type lora \
        --model_dir ./fine_tuned_model_run2 \
        --output_name run2

    python run_benchmark.py \
        --model_type lora \
        --model_dir ./fine_tuned_model_run1 \
        --output_name run1

    # Base Mavkif model, no adapter
    python run_benchmark.py \
        --model_type base \
        --output_name mavkif_base

Outputs:
    benchmark/results/predictions_<output_name>.csv
        columns: urdu_sentence, gold_roman_urdu, category, prediction
"""

import argparse
import csv
import os
import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from transformers.models.m2m_100.modeling_m2m_100 import M2M100Model
from peft import PeftModel

BASE_MODEL_ID     = "Mavkif/m2m100_rup_ur_to_rur"
TOKENIZER_ID      = "Mavkif/m2m100_rup_tokenizer_both"
TGT_LANG_TOKEN_ID = 128105   # __roman-ur__
SRC_LANG          = "ur"

BENCHMARK_CSV = "benchmark/benchmark_dataset.csv"
RESULTS_DIR   = "benchmark/results"


class PatchedM2M100Model(M2M100Model):
    """Fixes decoder_input_ids / decoder_inputs_embeds conflict in .generate()."""
    def forward(self, *args, **kwargs):
        if kwargs.get("decoder_inputs_embeds") is not None:
            kwargs.pop("decoder_input_ids", None)
        return super().forward(*args, **kwargs)


def patch_model(model):
    original = model.model
    patched = PatchedM2M100Model(model.config)
    patched.load_state_dict(original.state_dict())
    patched.to(next(original.parameters()).dtype)
    patched.to(next(original.parameters()).device)
    del model.model
    model.model = patched
    torch.cuda.empty_cache()
    return model


def load_model(model_type: str, model_dir: str, device: torch.device):
    print(f"[run_benchmark] Loading base model '{BASE_MODEL_ID}' ...")
    base_model = M2M100ForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID, torch_dtype=torch.float16
    )
    base_model.config.forced_bos_token_id = None
    base_model.generation_config.forced_bos_token_id = TGT_LANG_TOKEN_ID

    print("[run_benchmark] Applying PatchedM2M100Model ...")
    base_model = patch_model(base_model)

    if model_type == "lora":
        if not model_dir:
            raise ValueError("--model_dir is required when --model_type lora")
        print(f"[run_benchmark] Loading LoRA adapters from '{model_dir}' ...")
        model = PeftModel.from_pretrained(base_model, model_dir, torch_dtype=torch.float16)
        model = model.merge_and_unload()
    elif model_type == "base":
        print("[run_benchmark] Using base model, no adapter.")
        model = base_model
    else:
        raise ValueError(f"Unknown --model_type: {model_type}")

    model.to(device)
    model.eval()

    tokenizer_source = model_dir if (model_type == "lora" and model_dir) else TOKENIZER_ID
    print(f"[run_benchmark] Loading tokenizer from '{tokenizer_source}' ...")
    try:
        tokenizer = M2M100Tokenizer.from_pretrained(tokenizer_source)
    except Exception:
        print(f"  [warning] Falling back to '{TOKENIZER_ID}' from Hub.")
        tokenizer = M2M100Tokenizer.from_pretrained(TOKENIZER_ID)
    tokenizer.src_lang = SRC_LANG

    return model, tokenizer


def generate_batch(sentences, model, tokenizer, device, max_new_tokens=128, num_beams=4):
    inputs = tokenizer(
        sentences, return_tensors="pt", padding=True, truncation=True, max_length=128
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            forced_bos_token_id=TGT_LANG_TOKEN_ID,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
        )

    decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return [s.strip() for s in decoded]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=["lora", "base"], required=True)
    parser.add_argument("--model_dir", default=None, help="Path to LoRA adapter dir (required if model_type=lora)")
    parser.add_argument("--output_name", required=True, help="Name used for the output predictions file")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run_benchmark] Using device: {device}")

    with open(BENCHMARK_CSV, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"[run_benchmark] Loaded {len(rows)} benchmark sentences from '{BENCHMARK_CSV}'.")

    model, tokenizer = load_model(args.model_type, args.model_dir, device)

    sentences = [r["urdu_sentence"] for r in rows]
    predictions = []
    for i in range(0, len(sentences), args.batch_size):
        batch = sentences[i:i + args.batch_size]
        outputs = generate_batch(batch, model, tokenizer, device, num_beams=args.num_beams)
        predictions.extend(outputs)
        print(f"[run_benchmark] {min(i + args.batch_size, len(sentences))}/{len(sentences)} done")

    for r, pred in zip(rows, predictions):
        r["prediction"] = pred

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"predictions_{args.output_name}.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=["urdu_sentence", "gold_roman_urdu", "category", "prediction"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[run_benchmark] Done. Predictions saved to '{out_path}'.")


if __name__ == "__main__":
    main()
