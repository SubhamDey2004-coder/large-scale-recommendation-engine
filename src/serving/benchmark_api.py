import numpy as np
import csv
import statistics
import time
from pathlib import Path

import requests

URL = "http://127.0.0.1:8000/recommend/2262"
REQUESTS_COUNT = 20

latencies = []
results = []

# Warm up the API before measuring steady-state latency.
print("Warming up API...")

try:
    response = requests.get(URL, params={"top_k": 5}, timeout=60)
    response.raise_for_status()
    print("Warm-up successful.\n")
except requests.RequestException as exc:
    raise SystemExit(f"API warm-up failed: {exc}")

for i in range(REQUESTS_COUNT):
    start = time.perf_counter()

    try:
        response = requests.get(
            URL,
            params={"top_k": 5},
            timeout=60,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        count = 0
        if response.ok:
            try:
                count = response.json().get("count", 0)
            except ValueError:
                count = 0

        results.append({
            "request": i + 1,
            "status_code": response.status_code,
            "latency_ms": round(elapsed_ms, 2),
            "recommendations": count,
        })

        if not response.ok:
            print(f"Request {i + 1}: HTTP {response.status_code}")
            continue

        latencies.append(elapsed_ms)

        print(
            f"Request {i + 1:02d}: "
            f"{elapsed_ms:.1f} ms | "
            f"{count} recommendations"
        )

    except requests.RequestException as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000

        results.append({
            "request": i + 1,
            "status_code": "FAILED",
            "latency_ms": round(elapsed_ms, 2),
            "recommendations": 0,
        })

        print(f"Request {i + 1}: FAILED - {exc}")

if latencies:
    print("\n--- Benchmark Summary ---")
    print(f"Successful requests: {len(latencies)}/{REQUESTS_COUNT}")
    print(f"Average latency: {statistics.mean(latencies):.2f} ms")
    print(f"Median latency: {statistics.median(latencies):.2f} ms")
    print(f"Maximum latency: {max(latencies):.2f} ms")
    print(f"P95 latency: {np.percentile(latencies, 95):.2f} ms")

output_path = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "processed"
    / "api_benchmark.csv"
)

output_path.parent.mkdir(parents=True, exist_ok=True)

with output_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "request",
            "status_code",
            "latency_ms",
            "recommendations",
        ],
    )
    writer.writeheader()
    writer.writerows(results)

print(f"\nBenchmark saved to: {output_path}")
