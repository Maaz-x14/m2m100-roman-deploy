"""
model.py
--------
Owns the model lifecycle:
  - load_model()   — loads base model, applies patch, merges LoRA adapters
  - warmup()       — runs dummy inference calls so the first real request
                     doesn't pay JIT / cuDNN kernel compilation latency
  - transliterate() — the actual inference function, reused by every request

Nothing in this file imports FastAPI. It is pure PyTorch + HuggingFace.

Bug history carried forward from inference.py:
  #3  PatchedM2M100Model — drops decoder_input_ids when decoder_inputs_embeds
      is already present; required for .generate() to work in transformers 5.x
  #5  forced_bos_token_id must go on generation_config, not model.config
  #7  del model.model before assigning patched copy (prevents double VRAM usage)
"""

import logging
import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from transformers.models.m2m_100.modeling_m2m_100 import M2M100Model
from peft import PeftModel

from app import config

logger = logging.getLogger(__name__)

# ── Singleton state ───────────────────────────────────────────────────────────
# Populated once by load_model(), reused by every transliterate() call.

_model: M2M100ForConditionalGeneration | None = None
_tokenizer: M2M100Tokenizer | None = None
_device: torch.device | None = None
_model_ready: bool = False

# ── Warmup sentences ─────────────────────────────────────────────────────────
# Short, varied Urdu sentences covering different phonetic patterns.
# Chosen to exercise both common vocab and the loanword domain the model
# was fine-tuned to fix.

_WARMUP_SENTENCES = [
    "وہ گھر پر نہیں ہے",          # general conversational
    "اس کا لیپ ٹاپ کریش ہو گیا",  # loanword (laptop / crash)
    "آپ کیسے ہیں؟",               # greeting
]

# ── Patch — bug #3 ────────────────────────────────────────────────────────────

class PatchedM2M100Model(M2M100Model):
    """
    M2M100Model.forward() in transformers 5.x pre-computes
    decoder_inputs_embeds from decoder_input_ids, then passes BOTH to
    M2M100Decoder → ValueError.

    Fix: strip decoder_input_ids when decoder_inputs_embeds is already present.
    This covers the .generate() autoregressive path.
    """
    def forward(self, *args, **kwargs):
        if kwargs.get("decoder_inputs_embeds") is not None:
            kwargs.pop("decoder_input_ids", None)
        return super().forward(*args, **kwargs)


def _patch_model(
    model: M2M100ForConditionalGeneration,
) -> M2M100ForConditionalGeneration:
    """
    Swaps model.model in-place with PatchedM2M100Model.
    del before assign prevents double VRAM usage — bug #7.
    """
    original = model.model
    patched  = PatchedM2M100Model(model.config)
    patched.load_state_dict(original.state_dict())
    patched.to(next(original.parameters()).dtype)
    patched.to(next(original.parameters()).device)
    del model.model          # free original BEFORE assigning — bug #7
    model.model = patched
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model

# ── Public API ────────────────────────────────────────────────────────────────

def load_model() -> None:
    """
    Load base model, apply PatchedM2M100Model, merge LoRA adapters, load
    tokenizer. Populates module-level singletons. Called once at startup.

    Loading order:
      1. Load frozen base model in fp16
      2. Null out model.config.forced_bos_token_id (bug #5)
      3. Set generation_config.forced_bos_token_id = 128105 (__roman-ur__)
      4. Apply PatchedM2M100Model (bug #3)
      5. Wrap with PeftModel to attach LoRA adapters
      6. merge_and_unload() — bakes adapters into base weights, drops PEFT
         overhead, fastest possible inference
      7. Load tokenizer (from MODEL_DIR, falls back to Hub)
    """
    global _model, _tokenizer, _device, _model_ready

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", _device)

    logger.info("Loading base model '%s' ...", config.BASE_MODEL_ID)
    base_model = M2M100ForConditionalGeneration.from_pretrained(
        config.BASE_MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    # bug #5: forced_bos_token_id is only respected on generation_config in
    # transformers 5.x; setting it on model.config raises or is silently ignored.
    base_model.config.forced_bos_token_id = None
    base_model.generation_config.forced_bos_token_id = config.TGT_LANG_TOKEN_ID
    logger.info("forced_bos_token_id set to %d (__roman-ur__)", config.TGT_LANG_TOKEN_ID)

    logger.info("Applying PatchedM2M100Model (bug #3 fix) ...")
    base_model = _patch_model(base_model)

    logger.info("Loading LoRA adapters from '%s' ...", config.MODEL_DIR)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    peft_model = PeftModel.from_pretrained(
        base_model,
        config.MODEL_DIR,
        torch_dtype=dtype,
    )

    logger.info("Merging LoRA adapters into base weights ...")
    _model = peft_model.merge_and_unload()
    _model.to(_device)
    _model.eval()
    logger.info("Model ready on %s.", _device)

    # ── Tokenizer ─────────────────────────────────────────────────────────
    logger.info("Loading tokenizer from '%s' ...", config.MODEL_DIR)
    try:
        _tokenizer = M2M100Tokenizer.from_pretrained(config.MODEL_DIR)
    except Exception:
        logger.warning(
            "Tokenizer not found in '%s', falling back to Hub '%s'.",
            config.MODEL_DIR, config.TOKENIZER_ID,
        )
        _tokenizer = M2M100Tokenizer.from_pretrained(config.TOKENIZER_ID)

    _tokenizer.src_lang = config.SRC_LANG
    logger.info("Tokenizer loaded. src_lang='%s'.", config.SRC_LANG)

    _model_ready = True


def warmup() -> None:
    """
    Run dummy inference calls immediately after load_model() completes.

    Purpose: CUDA kernels and cuDNN convolution algorithms are compiled /
    selected on the first forward pass. Without warm-up the first real
    request pays this latency (can be several seconds). Subsequent calls
    hit cached paths and are fast.

    Uses the module-level _WARMUP_SENTENCES list (short, representative).
    Number of warm-up rounds is controlled by WARMUP_SENTENCES env var.
    """
    sentences = _WARMUP_SENTENCES[: config.WARMUP_SENTENCES]
    if not sentences:
        sentences = _WARMUP_SENTENCES[:1]

    logger.info("Running warm-up (%d sentence(s)) ...", len(sentences))
    try:
        result = transliterate(sentences)
        for urdu, roman in zip(sentences, result):
            logger.info("  warm-up: '%s' → '%s'", urdu, roman)
        logger.info("Warm-up complete. Service is ready.")
    except Exception as exc:
        logger.error("Warm-up failed: %s", exc, exc_info=True)
        raise RuntimeError("Model warm-up failed — server will not start.") from exc


def is_ready() -> bool:
    """Returns True once load_model() + warmup() have both completed."""
    return _model_ready


def transliterate(sentences: list[str]) -> list[str]:
    """
    Tokenizes a list of Urdu script sentences and generates Roman Urdu output.

    Args:
        sentences: Non-empty list of Urdu strings.

    Returns:
        List of Roman Urdu strings, same length and order as input.

    Raises:
        RuntimeError: If called before load_model() + warmup() complete.
        ValueError:   If sentences is empty.
    """
    if not _model_ready:
        raise RuntimeError("Model is not loaded yet.")
    if not sentences:
        raise ValueError("sentences list must not be empty.")

    inputs = _tokenizer(
        sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config.MAX_SRC_LEN,
    ).to(_device)

    # Detect silent truncation — warn per sentence so it's visible in logs
    # rather than silently dropping the tail of long inputs.
    for i, sent in enumerate(sentences):
        full_len = len(_tokenizer(sent, truncation=False)["input_ids"])
        if full_len > config.MAX_SRC_LEN:
            logger.warning(
                "Input truncated: %d tokens > MAX_SRC_LEN=%d. "
                "Sentence (first 60 chars): '%s'",
                full_len, config.MAX_SRC_LEN, sent[:60],
            )

    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            forced_bos_token_id=config.TGT_LANG_TOKEN_ID,
            max_new_tokens=config.MAX_NEW_TOKENS,
            num_beams=config.NUM_BEAMS,
            early_stopping=True,
        )

    decoded = _tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return [s.strip() for s in decoded]