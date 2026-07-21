"""
diagnose_to_thread.py
----------------------
WHY THIS EXISTS:
diagnose_overhead.py showed the real /romanize path spending ~368ms on
"model + thread" work, versus Phase 1's ~150-160ms for the same batch
size calling transliterate() directly, no thread hop. The leading
hypothesis is that asyncio.to_thread() itself — used in batcher.py to
keep the event loop free while generate() runs — is adding real cost,
possibly from CUDA context/stream handling across threads, possibly from
generic Python thread-pool scheduling.

This script isolates JUST that question: same process, same loaded model,
same input, same batch size — call transliterate() two ways back to back:

  (a) DIRECT — exactly like Phase 1 did. No thread hop.
  (b) VIA asyncio.to_thread() — exactly like batcher.py's _run_batch() does.

No HTTP, no FastAPI, no batcher queue, no Colab network round trip.
If (b) is meaningfully slower than (a), asyncio.to_thread() (or what
happens to CUDA when crossing threads) is the real cost, and the fix
needs to happen in batcher.py's execution strategy. If (a) and (b) are
close, the overhead is coming from somewhere else in the real server
path (worth then checking uvicorn's own thread pool usage, or the
queue/Future mechanics in batcher.py).

USAGE:
    Run from the repo root (same place you run bench_phase1.py):
        python diagnose_to_thread.py

    No server needs to be running — this loads the model itself, like
    Phase 1 did.
"""

import asyncio
import statistics
import time

from app import model as model_module
from app import config

N_SAMPLES = 20
WARMUP_DISCARD = 3
TEST_SENTENCE = "آج موسم بہت اچھا ہے چلو باہر چلتے ہیں اور کچھ وقت گزارتے ہیں"
BATCH_SIZE = 1  # matches the concurrency=1 case from the sweep where the gap was found


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
    print(f"{label:<35} p50={percentile(samples,50)*1000:.1f}ms  "
          f"p95={percentile(samples,95)*1000:.1f}ms  "
          f"mean={statistics.mean(samples)*1000:.1f}ms")


def run_direct(batch: list[str]) -> float:
    """Direct call, no thread hop — same as Phase 1."""
    t0 = time.perf_counter()
    model_module.transliterate(batch)
    return time.perf_counter() - t0


async def run_via_to_thread(batch: list[str]) -> float:
    """Same call, wrapped in asyncio.to_thread() — same as batcher.py's _run_batch()."""
    t0 = time.perf_counter()
    await asyncio.to_thread(model_module.transliterate, batch)
    return time.perf_counter() - t0


async def main():
    print("=" * 70)
    print("Isolating asyncio.to_thread() overhead — no HTTP, no FastAPI, no batcher")
    print("=" * 70)

    print("\nLoading model...")
    model_module.load_model()
    model_module.warmup()
    device = str(model_module._device)
    print(f"Device: {device}\n")

    batch = [TEST_SENTENCE] * BATCH_SIZE

    # ── Direct calls ─────────────────────────────────────────────────────
    direct_samples = []
    print(f"Running {N_SAMPLES} DIRECT calls (batch_size={BATCH_SIZE})...")
    for i in range(N_SAMPLES):
        t = run_direct(batch)
        if i >= WARMUP_DISCARD:
            direct_samples.append(t)

    # ── to_thread calls ──────────────────────────────────────────────────
    to_thread_samples = []
    print(f"Running {N_SAMPLES} asyncio.to_thread() calls (batch_size={BATCH_SIZE})...")
    for i in range(N_SAMPLES):
        t = await run_via_to_thread(batch)
        if i >= WARMUP_DISCARD:
            to_thread_samples.append(t)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    summarize("Direct call (no thread hop)", direct_samples)
    summarize("Via asyncio.to_thread()", to_thread_samples)

    direct_mean = statistics.mean(direct_samples)
    to_thread_mean = statistics.mean(to_thread_samples)
    diff_ms = (to_thread_mean - direct_mean) * 1000
    diff_pct = ((to_thread_mean / direct_mean) - 1) * 100

    print(f"\nDifference: {diff_ms:+.1f}ms ({diff_pct:+.1f}%)")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    if abs(diff_ms) < 20:
        print("  Difference is small (<20ms) — asyncio.to_thread() itself is NOT")
        print("  the main source of the ~200ms overhead seen in the real server path.")
        print("  The overhead is likely coming from elsewhere: uvicorn's own thread")
        print("  pool usage, the batcher's queue/Future mechanics, or something in")
        print("  FastAPI's request handling not captured by this isolated test.")
    elif diff_ms > 0:
        print(f"  asyncio.to_thread() adds ~{diff_ms:.0f}ms per call ({diff_pct:.0f}% slower).")
        print("  This IS a meaningful chunk of the overhead. Likely cause: CUDA")
        print("  context/stream handling when generate() runs on a different")
        print("  thread than the one that initialized CUDA, or default thread-pool")
        print("  scheduling latency. Worth checking if a dedicated single-thread")
        print("  executor (instead of the default pool) reduces this.")
    else:
        print("  to_thread() was actually FASTER — unexpected, worth re-running to confirm.")


if __name__ == "__main__":
    asyncio.run(main())
