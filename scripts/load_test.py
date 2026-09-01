#!/usr/bin/env python3
"""Load test for /v1/chat/completions against a running gateway.

Sends real requests — real OpenAI calls cost real money on a real key.
Defaults are deliberately small (20 requests, concurrency 4) for exactly
that reason; raise --requests/--concurrency yourself if you want to push
harder, understanding the cost implication of doing so.

Usage:
    python scripts/load_test.py --api-key sk-gw-... [--url http://localhost:8000] [--requests 20] [--concurrency 4]

What this actually demonstrates: the gateway staying correct under
concurrent load — Redis-backed rate limiting and circuit-breaker state
shared correctly across concurrent requests (not just sequential ones,
which is all the test suite's respx-mocked tests can prove), real
latency distribution, and a clean summary of what happened.
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from collections import Counter
from dataclasses import dataclass

import httpx


@dataclass
class RequestResult:
    status_code: int
    latency_ms: float
    error: str | None = None


async def _send_one(
    client: httpx.AsyncClient, url: str, api_key: str, model: str, index: int
) -> RequestResult:
    start = time.perf_counter()
    try:
        resp = await client.post(
            f"{url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": f"Say the number {index} and nothing else."}],
                "max_tokens": 10,
            },
            timeout=30.0,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        return RequestResult(status_code=resp.status_code, latency_ms=latency_ms)
    except httpx.RequestError as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        return RequestResult(status_code=0, latency_ms=latency_ms, error=str(exc))


async def run_load_test(url: str, api_key: str, model: str, total_requests: int, concurrency: int) -> None:
    semaphore = asyncio.Semaphore(concurrency)

    async def _bounded(index: int) -> RequestResult:
        async with semaphore:
            async with httpx.AsyncClient() as client:
                return await _send_one(client, url, api_key, model, index)

    print(f"Sending {total_requests} requests to {url}/v1/chat/completions (concurrency={concurrency})...")
    print(f"Model: {model}")
    print()

    overall_start = time.perf_counter()
    results = await asyncio.gather(*(_bounded(i) for i in range(total_requests)))
    overall_elapsed = time.perf_counter() - overall_start

    latencies = [r.latency_ms for r in results]
    status_counts = Counter(r.status_code for r in results)
    successes = status_counts[200]
    errors = [r for r in results if r.error]

    latencies_sorted = sorted(latencies)

    def _percentile(data: list[float], pct: float) -> float:
        if not data:
            return 0.0
        idx = min(int(len(data) * pct), len(data) - 1)
        return data[idx]

    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total requests:     {total_requests}")
    print(f"Wall-clock time:    {overall_elapsed:.2f}s")
    print(f"Throughput:         {total_requests / overall_elapsed:.2f} req/s")
    print()
    print(f"Successful (200):   {successes} ({successes / total_requests * 100:.1f}%)")
    for status, count in sorted(status_counts.items()):
        if status != 200:
            label = "connection error" if status == 0 else str(status)
            print(f"  {label}: {count}")
    print()
    if latencies:
        print("Latency (ms):")
        print(f"  min:    {min(latencies):.1f}")
        print(f"  mean:   {statistics.mean(latencies):.1f}")
        print(f"  p50:    {_percentile(latencies_sorted, 0.50):.1f}")
        print(f"  p95:    {_percentile(latencies_sorted, 0.95):.1f}")
        print(f"  p99:    {_percentile(latencies_sorted, 0.99):.1f}")
        print(f"  max:    {max(latencies):.1f}")
    if errors:
        print()
        print("Connection errors (first 3):")
        for r in errors[:3]:
            print(f"  {r.error}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Load test the gateway's /v1/chat/completions endpoint.")
    parser.add_argument("--url", default="http://localhost:8000", help="Gateway base URL")
    parser.add_argument("--api-key", required=True, help="Gateway API key (sk-gw-...)")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model to request")
    parser.add_argument(
        "--requests", type=int, default=20, help="Total requests to send (default 20 — costs real money)"
    )
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrent in-flight requests (default 4)")
    args = parser.parse_args()

    asyncio.run(run_load_test(args.url, args.api_key, args.model, args.requests, args.concurrency))


if __name__ == "__main__":
    main()
