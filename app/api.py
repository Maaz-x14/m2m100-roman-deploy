"""
api.py
------
FastAPI router with two endpoints:

  POST /romanize  — Urdu script → Roman Urdu transliteration
  GET  /health    — lightweight liveness + readiness probe

The root path (/) is intentionally left unmounted.
All model interaction goes through app.model; this file only handles
HTTP concerns (validation, error translation, response shaping).
"""

import logging
import time

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from app import model as model_module

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Request / Response schemas ────────────────────────────────────────────────

class RomanizeRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("text must not be empty or whitespace-only.")
        return v.strip()


class RomanizeResponse(BaseModel):
    romanized_text: str


class HealthResponse(BaseModel):
    status: str          # "ok" | "unavailable"
    model_ready: bool
    device: str

# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/romanize",
    response_model=RomanizeResponse,
    summary="Transliterate Urdu script to Roman Urdu",
    description=(
        "Accepts a single Urdu string in Arabic script and returns its "
        "phonetic Roman Urdu representation produced by the fine-tuned "
        "M2M100 LoRA model."
    ),
)
async def romanize(request: RomanizeRequest) -> RomanizeResponse:
    """
    POST /romanize

    Request body:
        { "text": "آپ کیسے ہیں؟" }

    Response:
        { "romanized_text": "Aap kaise hain?" }

    HTTP status codes:
        200 — success
        400 — blank / invalid input (caught by Pydantic validator)
        503 — model not ready
        500 — unexpected inference failure
    """
    if not model_module.is_ready():
        logger.warning("/romanize called before model is ready")
        return JSONResponse(
            status_code=503,
            content={"detail": "Model is not ready yet. Retry in a moment."},
        )

    logger.info("/romanize  input='%s'", request.text[:80])
    t0 = time.perf_counter()

    try:
        results = model_module.transliterate([request.text])
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
    romanized = results[0]
    logger.info("/romanize  output='%s'  (%.0f ms)", romanized[:80], elapsed)

    return RomanizeResponse(romanized_text=romanized)


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
    """
    GET /health

    Response (ready):
        { "status": "ok", "model_ready": true, "device": "cuda" }

    Response (not ready):
        HTTP 503
        { "status": "unavailable", "model_ready": false, "device": "unknown" }
    """
    ready  = model_module.is_ready()
    device = str(model_module._device) if model_module._device else "unknown"

    if not ready:
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "model_ready": False, "device": device},
        )

    return HealthResponse(status="ok", model_ready=True, device=device)
