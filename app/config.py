"""
config.py
---------
All runtime configuration loaded from environment variables.
No values are hardcoded here — change behaviour by setting env vars,
never by editing source code.

Environment variables (with defaults):
  HOST                 0.0.0.0
  PORT                 2000
  MODEL_DIR            ./fine_tuned_model
  NUM_BEAMS            4
  MAX_NEW_TOKENS       128
  INFERENCE_BATCH_SIZE 8       (max sentences per generate() call)
  WARMUP_SENTENCES     3       (dummy calls on startup)
  LOG_LEVEL            INFO
"""

import os

# ── Server ────────────────────────────────────────────────────────────────────
HOST: str = os.environ.get("HOST", "0.0.0.0")
PORT: int = int(os.environ.get("PORT", "2000"))

# ── Model ─────────────────────────────────────────────────────────────────────
BASE_MODEL_ID: str     = "Mavkif/m2m100_rup_ur_to_rur"
TOKENIZER_ID: str      = "Mavkif/m2m100_rup_tokenizer_both"
MODEL_DIR: str         = os.environ.get("MODEL_DIR", "./fine_tuned_model")
TGT_LANG_TOKEN_ID: int = 128105   # __roman-ur__
SRC_LANG: str          = "ur"

# ── Inference ─────────────────────────────────────────────────────────────────
NUM_BEAMS: int            = int(os.environ.get("NUM_BEAMS", "4"))
MAX_SRC_LEN: int          = int(os.environ.get("MAX_SRC_LEN", "256"))
MAX_NEW_TOKENS: int       = int(os.environ.get("MAX_NEW_TOKENS", "256"))
INFERENCE_BATCH_SIZE: int = int(os.environ.get("INFERENCE_BATCH_SIZE", "8"))
MAX_BATCH_SIZE: int       = INFERENCE_BATCH_SIZE  # reuse existing config value
MAX_WAIT_MS: int          = int(os.environ.get("MAX_WAIT_MS", "50"))


# ── Startup warm-up ───────────────────────────────────────────────────────────
WARMUP_SENTENCES: int = int(os.environ.get("WARMUP_SENTENCES", "3"))

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO").upper()