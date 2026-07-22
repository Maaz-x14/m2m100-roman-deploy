"""
run_sweep.py
------------
Automates the full MAX_BATCH_SIZE x MAX_WAIT_MS sweep:
  1. For each combination, kills any previous server, starts a fresh one
     with the right INFERENCE_BATCH_SIZE / MAX_WAIT_MS env vars.
  2. Waits for /health to report ready (polls, doesn't just sleep blindly).
  3. Rewrites CONFIG_LABEL inside bench_phase2.py to match this combination
     — you never need to hand-edit that file.
  4. Runs bench_phase2.py as a subprocess, streaming its output live.
  5. Kills the server, moves to the next combination.

At the end, prints a combined summary table across all 12 runs, parsed
from the per-combination CSVs each bench_phase2.py run produces
(phase2_results_<CONFIG_LABEL>.csv).

Usage:
    python run_sweep.py

Requires bench_phase2.py to be in the same directory, and the server
startable via `python -m uvicorn app.main:app` from this directory
(i.e. run this from the repo root, same place you'd normally start
the server from).
"""

import csv
import glob
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx

HOST = "0.0.0.0"
PORT = 2000
HEALTH_URL = f"http://localhost:{PORT}/health"
STARTUP_TIMEOUT_S = 60
STARTUP_POLL_INTERVAL_S = 2

BATCH_SIZES = [8, 16, 32, 48, 64]
WAIT_MSES = [30, 50, 70, 90]

BENCH_SCRIPT = "bench_phase2.py"


def wait_for_health(timeout_s: int) -> bool:
    """Polls /health until it returns 200, or timeout. Faster/safer than a blind sleep."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            resp = httpx.get(HEALTH_URL, timeout=3.0)
            if resp.status_code == 200 and resp.json().get("model_ready"):
                return True
        except Exception:
            pass
        time.sleep(STARTUP_POLL_INTERVAL_S)
    return False


def start_server(batch_size: int, wait_ms: int) -> subprocess.Popen:
    env = os.environ.copy()
    env["INFERENCE_BATCH_SIZE"] = str(batch_size)
    env["MAX_WAIT_MS"] = str(wait_ms)

    log_path = f"server_batch{batch_size}_wait{wait_ms}.log"
    proc = subprocess.Popen(
        ["python", "-m", "uvicorn", "app.main:app", "--host", HOST, "--port", str(PORT)],
        stdout=open(log_path, "w"), stderr=subprocess.STDOUT,
        env=env,
    )
    return proc


def stop_server(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)
    time.sleep(3)  # let the port fully release before starting the next server


def set_config_label(label: str) -> None:
    """Rewrites the CONFIG_LABEL line inside bench_phase2.py in place."""
    path = Path(BENCH_SCRIPT)
    content = path.read_text()
    new_content, n = re.subn(
        r'^CONFIG_LABEL = ".*?"',
        f'CONFIG_LABEL = "{label}"',
        content,
        count=1,
        flags=re.MULTILINE,
    )
    if n == 0:
        raise RuntimeError(
            f"Could not find CONFIG_LABEL line in {BENCH_SCRIPT} — "
            "check the file still has a line matching CONFIG_LABEL = \"...\""
        )
    path.write_text(new_content)


def run_benchmark() -> bool:
    """Runs bench_phase2.py as a subprocess, streaming output live. Returns True on success."""
    result = subprocess.run([sys.executable, BENCH_SCRIPT])
    return result.returncode == 0


def summarize_all_results() -> None:
    """After the full sweep, reads every phase2_results_*.csv and prints a combined table."""
    # Measured separately via diagnose_overhead.py — pure HTTP round trip to a
    # no-op /ping endpoint, no model involved. Negligible but subtracted here
    # so "model-only" numbers don't require manual arithmetic afterward.
    HTTP_OVERHEAD_S = 0.0018

    csv_files = sorted(glob.glob("phase2_results_*.csv"))
    if not csv_files:
        print("No result CSVs found to summarize.")
        return

    print("\n" + "=" * 110)
    print("SWEEP SUMMARY — all combinations (model-only = full latency minus ~1.8ms measured HTTP overhead)")
    print("=" * 110)
    print(f"{'Config':<20} {'Concurrency':<12} {'p50 (s)':<10} {'p95 (s)':<10} "
          f"{'Model p50':<11} {'Model p95':<11}")
    print("-" * 110)

    ranked = []
    for csv_path in csv_files:
        rows_by_concurrency: dict[int, list[float]] = {}
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            config_label = None
            for row in reader:
                config_label = row["config_label"]
                c = int(row["concurrency"])
                rows_by_concurrency.setdefault(c, []).append(float(row["request_latency_s"]))

        all_model_p95 = []
        for concurrency in sorted(rows_by_concurrency.keys()):
            latencies = sorted(rows_by_concurrency[concurrency])
            p50 = latencies[int(len(latencies) * 0.50)]
            p95 = latencies[min(int(len(latencies) * 0.95), len(latencies) - 1)]
            model_p50 = max(0.0, p50 - HTTP_OVERHEAD_S)
            model_p95 = max(0.0, p95 - HTTP_OVERHEAD_S)
            all_model_p95.append(model_p95)
            print(f"{config_label:<20} {concurrency:<12} {p50:<10.3f} {p95:<10.3f} "
                  f"{model_p50:<11.3f} {model_p95:<11.3f}")

        ranked.append((config_label, statistics.mean(all_model_p95), max(all_model_p95)))

    print("=" * 110)

    print("\n" + "=" * 70)
    print("RANKED BY AVERAGE MODEL-ONLY p95 ACROSS ALL CONCURRENCY LEVELS")
    print("(fixed rule — lower is better, no HTTP overhead included)")
    print("=" * 70)
    ranked.sort(key=lambda x: x[1])
    for i, (label, avg_p95, max_p95) in enumerate(ranked, 1):
        marker = "  <-- BEST" if i == 1 else ""
        under_300ms = "PASS" if max_p95 < 0.3 else "FAIL"
        print(f"{i:<4}{label:<20} avg_model_p95={avg_p95:.3f}  max_model_p95={max_p95:.3f}  "
              f"[{under_300ms} vs 300ms]{marker}")
    print("=" * 70)


def main():
    combinations = [(b, w) for b in BATCH_SIZES for w in WAIT_MSES]
    print(f"Starting sweep: {len(combinations)} combinations "
          f"(batch sizes {BATCH_SIZES} x wait_ms {WAIT_MSES})\n")

    for i, (batch_size, wait_ms) in enumerate(combinations, start=1):
        label = f"batch={batch_size}_wait={wait_ms}"
        print(f"\n{'#' * 70}")
        print(f"# [{i}/{len(combinations)}] {label}")
        print(f"{'#' * 70}")

        server = start_server(batch_size, wait_ms)
        try:
            print("Waiting for server to become healthy...")
            if not wait_for_health(STARTUP_TIMEOUT_S):
                print(f"  ERROR: server did not become healthy within {STARTUP_TIMEOUT_S}s — "
                      f"skipping {label}. Check server_batch{batch_size}_wait{wait_ms}.log")
                continue
            print("  Server healthy.")

            set_config_label(label)
            print(f"  CONFIG_LABEL set to '{label}' in {BENCH_SCRIPT}")

            print(f"  Running benchmark...")
            success = run_benchmark()
            if not success:
                print(f"  WARNING: benchmark run for {label} exited with non-zero status.")

        finally:
            print("  Stopping server...")
            stop_server(server)

    summarize_all_results()
    print("\nSweep complete.")


if __name__ == "__main__":
    main()
