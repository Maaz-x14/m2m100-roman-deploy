"""
batcher.py
----------
Dynamic batching layer sitting between the FastAPI endpoint and model.transliterate().

WHY THIS EXISTS:
Phase 2 benchmarking showed the server processes concurrent requests serially —
workers=1 plus a blocking transliterate() call inside an async endpoint means
request N cannot start until request N-1 finishes, regardless of how many
requests are queued. Phase 1 benchmarking showed the model itself batches
efficiently (per-item latency drops sharply as batch size grows, VRAM cost is
trivial). This module connects the two: it collects concurrent incoming
requests into batches and runs them through transliterate() together, instead
of one at a time.

DESIGN — size-or-time batching:
  A background loop waits for the first request, then keeps collecting more
  until EITHER:
    - the batch reaches MAX_BATCH_SIZE, OR
    - MAX_WAIT_MS has elapsed since the first request in this batch arrived
  whichever happens first — then runs transliterate() once on the whole batch.

  This adapts to load: high concurrency fills batches fast (near-zero extra
  wait), low concurrency falls back close to today's per-request latency
  (bounded by MAX_WAIT_MS, not unbounded).

WHAT THIS DOES NOT CHANGE:
  model.py (load_model, warmup, transliterate, the PatchedM2M100Model fix,
  LoRA merge) is untouched. This is purely a serving-layer addition.

ERROR HANDLING (explicit tradeoff, not hidden):
  If transliterate() raises for a batch, ALL requests in that batch receive
  the same error — including ones that would have succeeded on their own.
  This matches today's single-request error behavior (errors are the
  exception, not the common case) but is a known limitation worth revisiting
  if batch sizes grow large enough that one bad input regularly takes out
  many unrelated requests.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field

from app import config
from app import model as model_module

logger = logging.getLogger(__name__)


@dataclass
class _QueuedRequest:
    text: str
    future: "asyncio.Future[str]"
    enqueued_at: float = field(default_factory=time.perf_counter)


class DynamicBatcher:
    """
    Owns the request queue and the background batching loop.
    One instance is created at app startup and shared across all requests.
    """

    def __init__(self, max_batch_size: int, max_wait_ms: int):
        self.max_batch_size = max_batch_size
        self.max_wait_s = max_wait_ms / 1000.0
        self._queue: asyncio.Queue[_QueuedRequest] = asyncio.Queue()
        self._loop_task: asyncio.Task | None = None
        self._running = False

    def start(self) -> None:
        """Starts the background batching loop. Call once at app startup."""
        if self._loop_task is not None:
            logger.warning("DynamicBatcher.start() called twice — ignoring.")
            return
        self._running = True
        self._loop_task = asyncio.create_task(self._batch_loop())
        logger.info(
            "DynamicBatcher started (max_batch_size=%d, max_wait_ms=%d)",
            self.max_batch_size, int(self.max_wait_s * 1000),
        )

    async def stop(self) -> None:
        """Stops the background loop cleanly. Call at app shutdown."""
        self._running = False
        if self._loop_task is not None:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None
        logger.info("DynamicBatcher stopped.")

    async def submit(self, text: str) -> str:
        """
        Public entry point — called by the /romanize endpoint.
        Enqueues the request and awaits its own result. From the caller's
        perspective this behaves exactly like calling transliterate([text])[0]
        directly, just with better throughput under concurrent load.
        """
        future: "asyncio.Future[str]" = asyncio.get_event_loop().create_future()
        await self._queue.put(_QueuedRequest(text=text, future=future))
        return await future

    async def _batch_loop(self) -> None:
        """
        Background loop: waits for the first request, then collects more
        (size-or-time rule) before running one transliterate() call for
        the whole batch.
        """
        while self._running:
            try:
                # Block until at least one request arrives.
                first = await self._queue.get()
            except asyncio.CancelledError:
                break

            batch = [first]
            deadline = time.perf_counter() + self.max_wait_s

            # Keep collecting until MAX_BATCH_SIZE or the wait window closes.
            while len(batch) < self.max_batch_size:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                    batch.append(item)
                except asyncio.TimeoutError:
                    break

            await self._run_batch(batch)

    async def _run_batch(self, batch: list[_QueuedRequest]) -> None:
        texts = [item.text for item in batch]
        wait_times = [time.perf_counter() - item.enqueued_at for item in batch]

        logger.debug(
            "Running batch of %d (max wait in batch: %.3fs)",
            len(batch), max(wait_times),
        )

        try:
            # transliterate() is synchronous/blocking (calls model.generate()).
            # Run it in a thread so it doesn't block the event loop while
            # running — otherwise new requests couldn't even be ENQUEUED
            # while a batch is being processed, defeating the purpose.
            results = await asyncio.to_thread(model_module.transliterate, texts)
        except Exception as exc:
            logger.error("Batch inference failed for %d requests: %s", len(batch), exc, exc_info=True)
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(exc)
            return

        for item, result in zip(batch, results):
            if not item.future.done():
                item.future.set_result(result)


# ── Module-level singleton, created at app startup ─────────────────────────
_batcher: DynamicBatcher | None = None


def init_batcher() -> None:
    global _batcher
    _batcher = DynamicBatcher(
        max_batch_size=config.MAX_BATCH_SIZE,
        max_wait_ms=config.MAX_WAIT_MS,
    )
    _batcher.start()


async def shutdown_batcher() -> None:
    global _batcher
    if _batcher is not None:
        await _batcher.stop()
        _batcher = None


def get_batcher() -> DynamicBatcher:
    if _batcher is None:
        raise RuntimeError("Batcher not initialized — init_batcher() must be called at startup.")
    return _batcher
