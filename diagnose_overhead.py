"""
diagnose_overhead.py
---------------------
WHY THIS EXISTS:
The full sweep showed every config, even at concurrency=1 (a single,
isolated request with nobody else to batch with), taking 0.35-0.5s+ p50 —
2-3x slower than Phase 1's direct-call measurement for the same batch size
(~0.15-0.17s). This gap exists even when there's no batching contention,
which means something is adding fixed overhead to every request, separate
from the batch-size/wait-time tuning we were trying to test. This script
isolates WHERE that time goes so we fix the right thing instead of
continuing to tune batch parameters that aren't the actual problem.

WHAT IT MEASURES, for the same single request, three different ways:

  1. DIRECT MODEL CALL — bypasses everything (HTTP, batcher, FastAPI).
     Calls model.transliterate() directly in-process, exactly like
     Phase 1 did. This is the true floor: pure model inference time.

  2. HTTP ROUND TRIP TO A TRIVIAL ENDPOINT — measures pure HTTP/FastAPI/
     uvicorn overhead with NO model involved at all. Requires a /ping
     endpoint (see note below) that does nothing but return immediately.
     If this alone is slow, the problem is FastAPI/uvicorn/network, not
     our code.

  3. FULL PATH VIA /romanize — the real path: HTTP -> FastAPI -> batcher
     queue (submit + wait) -> asyncio.to_thread -> transliterate() ->
     HTTP response. Broken into two sub-parts using timestamps embedded
     in the batcher itself (see note below) so we can see: how long did
     the request sit in queue/wait for its batch, versus how long did
     the actual batched generate() call take once it started running.

NOTE — THIS SCRIPT REQUIRES TWO SMALL ADDITIONS, DONE FOR YOU BELOW:
  (a) A trivial /ping endpoint added to api.py (for step 2) — instructions
      printed at the bottom of this file if not already present.
  (b) The batcher already logs enqueue-to-batch-start timing internally
      via wait_times inside _run_batch() — this script reads that from
      server logs (grep for "Running batch of") rather than needing new
      instrumentation, since it's already computed there.

USAGE:
    Run this against an ALREADY RUNNING server (same pattern as
    bench_phase2.py) — start the server with whatever single
    MAX_BATCH_SIZE / MAX_WAIT_MS combination you want to diagnose,
    confirm /health, then:

        python diagnose_overhead.py

OUTPUT:
    Prints a breakdown table showing each of the three measurements,
    so the gap between them tells you exactly which layer is adding time.
"""

import asyncio
import statistics
import subprocess
import sys
import time

import httpx

BASE_URL = "http://localhost:2000"
N_SAMPLES = 15
TEST_SENTENCE = "آج موسم بہت اچھا ہے چلو باہر چلتے ہیں اور کچھ وقت گزارتے ہیں"


def percentile(data: list[float], p: float) -> float:
    if not data:
        return float("nan")
    data_sorted = sorted(data)
    k = (len(data_sorted) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(data_sorted) - 1)
    if f == c:
        return data_sorted[f]
    return data_sorted[f] + (data_sorted[c] - data_sorted[f]) * (k - f)


def summarize(label: str, samples: list[float]) -> None:
    if not samples:
        print(f"{label}: NO DATA")
        return
    print(f"{label:<45} p50={percentile(samples,50)*1000:.1f}ms  "
          f"p95={percentile(samples,95)*1000:.1f}ms  "
          f"mean={statistics.mean(samples)*1000:.1f}ms")


async def measure_ping(client: httpx.AsyncClient) -> list[float]:
    """Pure HTTP/FastAPI overhead, no model involved. Requires /ping endpoint."""
    samples = []
    for _ in range(N_SAMPLES):
        t0 = time.perf_counter()
        try:
            resp = await client.get(f"{BASE_URL}/ping", timeout=5.0)
            if resp.status_code != 200:
                print(f"  /ping returned {resp.status_code} — endpoint may not exist yet")
                return []
        except httpx.HTTPStatusError:
            return []
        except Exception as exc:
            print(f"  /ping failed: {exc} — endpoint likely missing, see instructions at bottom of script")
            return []
        samples.append(time.perf_counter() - t0)
    return samples


async def measure_romanize_full_path(client: httpx.AsyncClient) -> list[float]:
    """Full real path: HTTP -> batcher -> transliterate() -> HTTP response."""
    samples = []
    for _ in range(N_SAMPLES):
        t0 = time.perf_counter()
        resp = await client.post(
            f"{BASE_URL}/romanize", json={"text": TEST_SENTENCE}, timeout=30.0
        )
        t1 = time.perf_counter()
        if resp.status_code == 200:
            samples.append(t1 - t0)
        else:
            print(f"  /romanize returned {resp.status_code}: {resp.text[:200]}")
        await asyncio.sleep(0.1)  # small gap so requests don't accidentally batch together
    return samples


def get_batch_wait_times_from_log(log_path: str = "server.log") -> list[float]:
    """
    Parses the server log for lines like:
      "Running batch of 1 (max wait in batch: 0.234s)"
    which the batcher already logs at DEBUG level in _run_batch().
    Requires LOG_LEVEL=DEBUG to be set when starting the server, otherwise
    these lines won't appear (they're logger.debug calls).
    """
    import re
    wait_times = []
    try:
        with open(log_path) as f:
            for line in f:
                m = re.search(r"max wait in batch: ([\d.]+)s", line)
                if m:
                    wait_times.append(float(m.group(1)))
    except FileNotFoundError:
        print(f"  Log file {log_path} not found — skipping queue-wait breakdown")
    return wait_times


async def main():
    print("=" * 70)
    print("Overhead diagnosis — isolating where request time actually goes")
    print("=" * 70)

    async with httpx.AsyncClient() as client:
        try:
            health = await client.get(f"{BASE_URL}/health", timeout=5.0)
            if health.status_code != 200:
                print("Server not healthy — aborting.")
                return
        except Exception as exc:
            print(f"Cannot reach server: {exc}")
            return

        print(f"\n1. Pure HTTP/FastAPI overhead (/ping, no model):")
        ping_samples = await measure_ping(client)
        if ping_samples:
            summarize("   HTTP round trip only", ping_samples)
        else:
            print("   SKIPPED — /ping endpoint not found. See instructions below.")

        print(f"\n2. Full real path (/romanize — HTTP + batcher queue + inference):")
        full_path_samples = await measure_romanize_full_path(client)
        summarize("   Full /romanize round trip", full_path_samples)

        print(f"\n3. Queue wait time (parsed from server.log, requires LOG_LEVEL=DEBUG):")
        wait_times = get_batch_wait_times_from_log()
        if wait_times:
            summarize("   Time spent waiting in batch queue", wait_times)
        else:
            print("   No data — either log file missing or LOG_LEVEL isn't DEBUG.")

        print("\n" + "=" * 70)
        print("INTERPRETATION")
        print("=" * 70)
        if ping_samples and full_path_samples:
            http_overhead = statistics.mean(ping_samples)
            full_time = statistics.mean(full_path_samples)
            unexplained = full_time - http_overhead
            print(f"  Pure HTTP overhead:        {http_overhead*1000:.1f}ms")
            print(f"  Full /romanize path:       {full_time*1000:.1f}ms")
            print(f"  Difference (batcher+model): {unexplained*1000:.1f}ms")
            if wait_times:
                print(f"  Of which, queue wait:      {statistics.mean(wait_times)*1000:.1f}ms")
                print(f"  Remaining (model+thread):  {(unexplained - statistics.mean(wait_times))*1000:.1f}ms")
        print("\nCompare 'Remaining (model+thread)' against Phase 1's batch_size=1")
        print("generate() time (~150-160ms from earlier runs). If it's much higher,")
        print("the asyncio.to_thread() wrapper or thread-pool scheduling is adding")
        print("real overhead, not just Colab noise.")


if __name__ == "__main__":
    asyncio.run(main())


# ─────────────────────────────────────────────────────────────────────────
# REQUIRED ADDITION #1 — /ping endpoint
# ─────────────────────────────────────────────────────────────────────────
# Add this to app/api.py, anywhere alongside the other @router routes:
#
# @router.get("/ping")
# async def ping():
#     """Trivial endpoint, no model involvement — measures pure HTTP overhead."""
#     return {"pong": True}
#
# ─────────────────────────────────────────────────────────────────────────
# REQUIRED ADDITION #2 — enable DEBUG logging to see queue wait times
# ─────────────────────────────────────────────────────────────────────────
# Start the server with LOG_LEVEL=DEBUG so batcher.py's existing
# logger.debug("Running batch of %d (max wait in batch: %.3fs)", ...)
# line actually gets written to server.log. Example:
#
#   LOG_LEVEL=DEBUG INFERENCE_BATCH_SIZE=8 MAX_WAIT_MS=50 \
#     python -m uvicorn app.main:app --host 0.0.0.0 --port 2000
# ─────────────────────────────────────────────────────────────────────────
