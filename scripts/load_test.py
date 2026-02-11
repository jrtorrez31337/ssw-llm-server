#!/usr/bin/env python3
"""SSW AI Gateway — Load Test

Hits both heavy and light models with concurrent requests,
measures throughput, latency percentiles, and error rates.

Usage:
    pip install httpx
    python load_test.py
    python load_test.py --heavy 10 --light 20 --rounds 3
"""

import argparse
import asyncio
import json
import statistics
import time

import httpx

DEFAULT_GATEWAY = "http://172.25.63.5:8000"

PROMPTS = [
    "You are a space station merchant. A customer asks about fuel prices. Reply in one sentence.",
    "You are a fleet commander. Summarize the tactical situation in two sentences.",
    "You are a bartender on a space station. Greet a new customer in one sentence.",
    "You are a bounty hunter. Describe your current target in two sentences.",
    "You are a navigation AI. Report the shortest route to sector G-7 in one sentence.",
    "You are a diplomat. Propose a ceasefire in two sentences.",
    "You are a pirate captain. Taunt an approaching patrol ship in one sentence.",
    "You are a mining foreman. Report today's ore yield in one sentence.",
]


async def send_request(
    client: httpx.AsyncClient,
    gateway: str,
    model: str,
    prompt: str,
    stream: bool = False,
) -> dict:
    """Send a single chat completion request, return timing + result info."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 64,
        "temperature": 0.7,
    }
    url = f"{gateway}/v1/chat/completions"
    t0 = time.monotonic()

    if stream:
        body["stream"] = True
        tokens = 0
        first_token_ms = None
        try:
            async with client.stream("POST", url, json=body) as resp:
                status = resp.status_code
                async for line in resp.aiter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        if first_token_ms is None:
                            first_token_ms = (time.monotonic() - t0) * 1000
                        chunk = json.loads(line[6:])
                        if chunk["choices"][0]["delta"].get("content"):
                            tokens += 1
        except Exception as e:
            return {"model": model, "status": 0, "error": str(e), "latency_ms": (time.monotonic() - t0) * 1000}

        return {
            "model": model,
            "status": status,
            "latency_ms": (time.monotonic() - t0) * 1000,
            "ttft_ms": first_token_ms,
            "tokens": tokens,
        }
    else:
        try:
            resp = await client.post(url, json=body)
            latency = (time.monotonic() - t0) * 1000
            data = resp.json()
            tokens = data.get("usage", {}).get("completion_tokens", 0)
            return {
                "model": model,
                "status": resp.status_code,
                "latency_ms": latency,
                "tokens": tokens,
            }
        except Exception as e:
            return {"model": model, "status": 0, "error": str(e), "latency_ms": (time.monotonic() - t0) * 1000}


async def run_round(
    gateway: str,
    n_heavy: int,
    n_light: int,
    stream: bool,
    round_num: int,
) -> list[dict]:
    """Fire n_heavy + n_light concurrent requests."""
    print(f"\n--- Round {round_num}: {n_heavy} heavy + {n_light} light ({'streaming' if stream else 'non-streaming'}) ---")

    async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=10.0)) as client:
        tasks = []
        for i in range(n_heavy):
            tasks.append(send_request(client, gateway, "heavy", PROMPTS[i % len(PROMPTS)], stream))
        for i in range(n_light):
            tasks.append(send_request(client, gateway, "light", PROMPTS[i % len(PROMPTS)], stream))

        t0 = time.monotonic()
        results = await asyncio.gather(*tasks)
        wall_time = (time.monotonic() - t0) * 1000

    print(f"    Wall time: {wall_time:.0f}ms")
    return list(results)


def print_stats(results: list[dict], label: str):
    """Print latency percentiles and error rate for a set of results."""
    if not results:
        return

    latencies = [r["latency_ms"] for r in results if r["status"] == 200]
    errors = [r for r in results if r["status"] != 200]
    tokens = sum(r.get("tokens", 0) for r in results)
    ttfts = [r["ttft_ms"] for r in results if r.get("ttft_ms") is not None]

    print(f"\n  [{label}] {len(results)} requests, {len(errors)} errors, {tokens} tokens generated")

    if latencies:
        latencies.sort()
        print(f"    Latency  min={latencies[0]:.0f}ms  p50={latencies[len(latencies)//2]:.0f}ms  "
              f"p95={latencies[int(len(latencies)*0.95)]:.0f}ms  p99={latencies[int(len(latencies)*0.99)]:.0f}ms  "
              f"max={latencies[-1]:.0f}ms")
        if len(latencies) > 1:
            print(f"    Mean: {statistics.mean(latencies):.0f}ms  Stdev: {statistics.stdev(latencies):.0f}ms")
        else:
            print(f"    Mean: {statistics.mean(latencies):.0f}ms")

    if ttfts:
        ttfts.sort()
        print(f"    TTFT     min={ttfts[0]:.0f}ms  p50={ttfts[len(ttfts)//2]:.0f}ms  max={ttfts[-1]:.0f}ms")

    if errors:
        for e in errors[:3]:
            err_msg = e.get("error") or "status %s" % e.get("status")
            print(f"    ERROR: {err_msg}")


async def main():
    parser = argparse.ArgumentParser(description="SSW AI Gateway Load Test")
    parser.add_argument("--gateway", default=DEFAULT_GATEWAY, help="Gateway URL")
    parser.add_argument("--heavy", type=int, default=5, help="Concurrent heavy requests per round")
    parser.add_argument("--light", type=int, default=10, help="Concurrent light requests per round")
    parser.add_argument("--rounds", type=int, default=3, help="Number of rounds")
    parser.add_argument("--stream", action="store_true", help="Use streaming mode")
    parser.add_argument("--ramp", action="store_true", help="Ramp up concurrency each round (1x, 2x, 3x...)")
    args = parser.parse_args()

    gateway = args.gateway

    print("SSW AI Gateway Load Test")
    print(f"Target: {gateway}")
    print(f"Config: {args.heavy} heavy + {args.light} light x {args.rounds} rounds")

    # Pre-flight check
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"{gateway}/health")
            health = r.json()
            healthy = sum(1 for v in health["workers"].values() if v == "healthy")
            print(f"Health: {health['status']} — {healthy}/{len(health['workers'])} workers up")
    except Exception as e:
        print(f"FATAL: Cannot reach gateway: {e}")
        return

    all_results: list[dict] = []

    for i in range(1, args.rounds + 1):
        multiplier = i if args.ramp else 1
        results = await run_round(
            gateway,
            args.heavy * multiplier,
            args.light * multiplier,
            args.stream,
            i,
        )
        all_results.extend(results)

        heavy_results = [r for r in results if r["model"] == "heavy"]
        light_results = [r for r in results if r["model"] == "light"]
        print_stats(heavy_results, "heavy")
        print_stats(light_results, "light")

    # Final summary
    print("\n" + "=" * 60)
    print("TOTAL SUMMARY")
    print("=" * 60)
    heavy_all = [r for r in all_results if r["model"] == "heavy"]
    light_all = [r for r in all_results if r["model"] == "light"]
    print_stats(heavy_all, "heavy")
    print_stats(light_all, "light")

    total_ok = sum(1 for r in all_results if r["status"] == 200)
    total_err = sum(1 for r in all_results if r["status"] != 200)
    print(f"\n  Total: {total_ok} succeeded, {total_err} failed out of {len(all_results)} requests")

    # Pull gateway metrics
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"{gateway}/metrics")
            print(f"\n  Gateway metrics: {json.dumps(r.json()['models'], indent=2)}")
    except Exception:
        pass


if __name__ == "__main__":
    asyncio.run(main())
