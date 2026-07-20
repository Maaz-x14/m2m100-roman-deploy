"""
main.py
-------
FastAPI application entry point.

Startup sequence (lifespan):
  1. Configure logging
  2. load_model()  — loads base model + LoRA adapters, merges weights
  3. warmup()      — dummy inference calls to prime CUDA / cuDNN kernels
  4. API begins accepting requests

Shutdown sequence:
  Lifespan context manager exits cleanly; Python GC + CUDA context teardown
  handles memory release. No explicit teardown needed for this service.

Root endpoint (/) is intentionally NOT mounted.
All routes live under the /app.api router.
"""

import logging
import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from app import config
from app import model as model_module
from app.api import router
from app import batcher as batcher_module

# ── Logging ───────────────────────────────────────────────────────────────────

def _configure_logging() -> None:
    log_format = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
    logging.basicConfig(
        stream=sys.stdout,
        level=getattr(logging, config.LOG_LEVEL, logging.INFO),
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Silence overly verbose HuggingFace / tokenizers loggers in production
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("tokenizers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("peft").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)

# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs model loading and warm-up BEFORE the server accepts any connections.
    FastAPI's lifespan protocol guarantees the server only starts serving
    after the `yield` — so all startup work completes first.

    Any exception here will abort startup cleanly (non-zero exit code),
    which is exactly what we want: fail fast if the model can't load.
    """
    _configure_logging()
    logger.info("=" * 60)
    logger.info("Romanize API starting up")
    logger.info("  HOST      : %s", config.HOST)
    logger.info("  PORT      : %d", config.PORT)
    logger.info("  MODEL_DIR : %s", config.MODEL_DIR)
    logger.info("  NUM_BEAMS : %d", config.NUM_BEAMS)
    logger.info("=" * 60)

    try:
        model_module.load_model()
        model_module.warmup()
        batcher_module.init_batcher()          # <-- ADD THIS LINE
    except Exception as exc:
        logger.critical("Startup failed: %s", exc, exc_info=True)
        raise

    logger.info("Romanize API is READY. Accepting requests.")
    logger.info("  POST http://%s:%d/romanize", config.HOST, config.PORT)
    logger.info("  GET  http://%s:%d/health",   config.HOST, config.PORT)

    yield  # ← server runs here

    await batcher_module.shutdown_batcher()    # <-- ADD THIS LINE
    logger.info("Romanize API shutting down.")

# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="Romanize API",
        description=(
            "Fine-tuned M2M100 LoRA model for Urdu script → Roman Urdu "
            "transliteration. Expose POST /romanize for inference."
        ),
        version="1.0.0",
        lifespan=lifespan,
        # Root (/) is intentionally not mounted.
        # docs available at /docs and /redoc.
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── Validation error handler ───────────────────────────────────────────
    # Pydantic validation errors (e.g. blank text) return 422 by default.
    # Override to return 400 with a clean message.
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request, exc):
        errors = exc.errors()
        # Extract the first meaningful message
        detail = errors[0].get("msg", str(exc)) if errors else str(exc)
        return JSONResponse(status_code=400, content={"detail": detail})

    app.include_router(router)
    return app


app = create_app()

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=config.HOST,
        port=config.PORT,
        log_config=None,     # use our own logging config, not uvicorn's default
        access_log=True,
    )
