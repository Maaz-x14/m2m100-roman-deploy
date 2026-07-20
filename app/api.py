"""
api.py
------
FastAPI router with two endpoints:

  POST /romanize  — Urdu script → Roman Urdu transliteration
  GET  /health    — lightweight liveness + readiness probe

text field accepts either a single string or a list of strings.
Response mirrors the input shape:
  - string in  → { "romanized_text": "..." }
  - list in    → { "romanized_text": ["...", "...", ...] }

The root path (/) is intentionally left unmounted.
All model interaction goes through app.model; this file only handles
HTTP concerns (validation, error translation, response shaping).
"""

import logging
import time
from typing import Union

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from app import model as model_module
from app import batcher as batcher_module


logger = logging.getLogger(__name__)

router = APIRouter()

# ── Request / Response schemas ────────────────────────────────────────────────

class RomanizeRequest(BaseModel):
    text: Union[str, list[str]]

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, v: Union[str, list[str]]) -> Union[str, list[str]]:
        if isinstance(v, str):
            if not v.strip():
                raise ValueError("text must not be empty or whitespace-only.")
            return v.strip()
        # list path
        if not v:
            raise ValueError("text list must not be empty.")
        cleaned = [s.strip() for s in v]
        blanks  = [i for i, s in enumerate(cleaned) if not s]
        if blanks:
            raise ValueError(f"text list has blank entries at index: {blanks}.")
        return cleaned


class RomanizeResponse(BaseModel):
    romanized_text: Union[str, list[str]]


class HealthResponse(BaseModel):
    status: str       # "ok" | "unavailable"
    model_ready: bool
    device: str

# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/romanize",
    response_model=RomanizeResponse,
    summary="Transliterate Urdu script to Roman Urdu",
    description=(
        "Accepts a single Urdu string or a list of strings in Arabic script "
        "and returns Roman Urdu. Response shape mirrors the input: "
        "string → string, list → list. Single-string requests are grouped "
        "into dynamic batches with other concurrent requests for throughput; "
        "list requests are processed as an explicit batch immediately."
    ),
)
async def romanize(request: RomanizeRequest) -> RomanizeResponse:
    """
    POST /romanize

    Single string:
        { "text": "آپ کیسے ہیں؟" }
        → { "romanized_text": "Aap kaise hain?" }

    List of strings:
        { "text": ["آپ کیسے ہیں؟", "وہ گھر پر نہیں ہے"] }
        → { "romanized_text": ["Aap kaise hain?", "woh ghar par nahi hai"] }

    HTTP status codes:
        200 — success
        400 — blank / invalid input
        503 — model not ready
        500 — unexpected inference failure
    """
    if not model_module.is_ready():
        logger.warning("/romanize called before model is ready")
        return JSONResponse(
            status_code=503,
            content={"detail": "Model is not ready yet. Retry in a moment."},
        )

    is_batch = isinstance(request.text, list)
    sentences = request.text if is_batch else [request.text]

    logger.info(
        "/romanize  %s  n=%d  preview='%s'",
        "batch" if is_batch else "single",
        len(sentences),
        sentences[0][:60],
    )
    t0 = time.perf_counter()

    try:
        if is_batch:
            # Explicit client-supplied batch — run directly, unchanged from
            # before. Bypasses the dynamic batcher (see note above).
            results = model_module.transliterate(sentences)
        else:
            # Single request — route through the dynamic batcher so it can
            # be grouped with other concurrent single requests.
            batcher = batcher_module.get_batcher()
            result = await batcher.submit(sentences[0])
            results = [result]
    except ValueError as exc:
        logger.warning("/romanize bad input: %s", exc)
        return JSONResponse(status_code=400, content={"detail": str(exc)})
    except Exception as exc:
        logger.error("/romanize inference error: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Inference failed. See server logs for details."},
        )

    elapsed = (time.perf_counter() - t0) * 1000
    logger.info("/romanize  done  n=%d  (%.0f ms)", len(results), elapsed)

    return RomanizeResponse(romanized_text=results if is_batch else results[0])



@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health and readiness check",
    description=(
        "Returns 200 when the model is loaded and warm-up is complete. "
        "Returns 503 if the model is still initialising."
    ),
)
async def health() -> HealthResponse:
    ready  = model_module.is_ready()
    device = str(model_module._device) if model_module._device else "unknown"

    if not ready:
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "model_ready": False, "device": device},
        )

    return HealthResponse(status="ok", model_ready=True, device=device)