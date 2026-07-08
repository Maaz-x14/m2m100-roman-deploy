#!/usr/bin/env bash
# start.sh
# --------
# Production startup script for the Romanize API.
# Run from the romanize_api/ directory:
#
#   chmod +x scripts/start.sh
#   ./scripts/start.sh
#
# Override any default via env var:
#   PORT=8080 ./scripts/start.sh

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-2000}"
MODEL_DIR="${MODEL_DIR:-./fine_tuned_model}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

echo "[start.sh] Starting Romanize API"
echo "  HOST      : $HOST"
echo "  PORT      : $PORT"
echo "  MODEL_DIR : $MODEL_DIR"

# ── Single GPU (mirrors train.py bug #10) ─────────────────────────────────────
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# ── Reduce CUDA allocator fragmentation ───────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ── Launch ────────────────────────────────────────────────────────────────────
# workers=1: model lives in VRAM and must not be forked after load.
exec uvicorn app.main:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers 1 \
    --no-access-log \
    --log-level "$(echo "$LOG_LEVEL" | tr '[:upper:]' '[:lower:]')"