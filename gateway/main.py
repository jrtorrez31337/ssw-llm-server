import asyncio
import json
import time
from contextlib import asynccontextmanager

import httpx
import redis.asyncio as redis
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from config import Settings, WorkerPool, build_pools, settings

# ---------------------------------------------------------------------------
# Globals (initialized in lifespan)
# ---------------------------------------------------------------------------
pools: dict[str, WorkerPool] = {}
http_client: httpx.AsyncClient = None  # type: ignore[assignment]
redis_client: redis.Redis = None  # type: ignore[assignment]
_health_task: asyncio.Task | None = None


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global pools, http_client, redis_client, _health_task

    pools = build_pools()
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(
            settings.worker_timeout, connect=settings.worker_connect_timeout
        ),
        limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
    )
    redis_client = redis.from_url(settings.redis_url, decode_responses=True)
    _health_task = asyncio.create_task(_health_loop())

    yield

    _health_task.cancel()
    await http_client.aclose()
    await redis_client.aclose()


app = FastAPI(title="SSW AI Gateway", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Health-check background loop
# ---------------------------------------------------------------------------
async def _check_worker(pool: WorkerPool, url: str):
    try:
        resp = await http_client.get(f"{url}/health", timeout=5.0)
        if resp.status_code == 200:
            pool.healthy.add(url)
        else:
            pool.healthy.discard(url)
    except Exception:
        pool.healthy.discard(url)


async def _health_loop():
    while True:
        tasks = []
        for pool in pools.values():
            for url in pool.all_urls():
                tasks.append(_check_worker(pool, url))
        await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.sleep(settings.health_check_interval)


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------
def _select_worker(model: str) -> tuple[WorkerPool, str]:
    """Pick a pool by model name and return the next healthy worker URL."""
    pool = pools.get(model)
    if pool is None:
        raise ValueError(f"Unknown model: {model!r}. Must be 'heavy' or 'light'.")
    url = pool.next()
    if url is None:
        raise RuntimeError(f"No healthy workers in pool {model!r}")
    return pool, url


async def _log_metric(model: str, latency_ms: float, status: int, tokens: int | None):
    """Push request metric to Redis list (best-effort, fire-and-forget)."""
    try:
        entry = json.dumps({
            "ts": time.time(),
            "model": model,
            "latency_ms": round(latency_ms, 1),
            "status": status,
            "tokens": tokens,
        })
        pipe = redis_client.pipeline(transaction=False)
        pipe.lpush("ssw:metrics", entry)
        pipe.ltrim("ssw:metrics", 0, settings.metrics_max_entries - 1)
        await pipe.execute()
    except Exception:
        pass  # metrics are best-effort


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    worker_status = {}
    for name, pool in pools.items():
        for url in pool.all_urls():
            worker_status[url] = "healthy" if url in pool.healthy else "unhealthy"
    all_healthy = all(v == "healthy" for v in worker_status.values())
    return JSONResponse(
        status_code=200 if all_healthy else 503,
        content={"status": "ok" if all_healthy else "degraded", "workers": worker_status},
    )


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "heavy", "object": "model", "owned_by": "ssw"},
            {"id": "light", "object": "model", "owned_by": "ssw"},
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model = body.get("model", "heavy")
    is_stream = body.get("stream", False)

    # Inject thinking suppression for light model
    if model == "light":
        body.setdefault("chat_template_kwargs", {})["enable_thinking"] = False

    # Select worker
    try:
        pool, worker_url = _select_worker(model)
    except (ValueError, RuntimeError) as e:
        return JSONResponse(status_code=503, content={"error": str(e)})

    target = f"{worker_url}/v1/chat/completions"
    t0 = time.monotonic()

    # --- Streaming ---
    if is_stream:
        try:
            worker_req = http_client.build_request("POST", target, json=body)
            worker_resp = await http_client.send(worker_req, stream=True)
        except Exception as exc:
            # Try failover to another healthy worker
            fallback_url = pool.next()
            if fallback_url and fallback_url != worker_url:
                try:
                    worker_req = http_client.build_request(
                        "POST", f"{fallback_url}/v1/chat/completions", json=body
                    )
                    worker_resp = await http_client.send(worker_req, stream=True)
                except Exception:
                    return JSONResponse(status_code=502, content={"error": f"All workers failed: {exc}"})
            else:
                return JSONResponse(status_code=502, content={"error": f"Worker unreachable: {exc}"})

        async def stream_generator():
            try:
                async for chunk in worker_resp.aiter_bytes():
                    yield chunk
            finally:
                await worker_resp.aclose()
                latency = (time.monotonic() - t0) * 1000
                await _log_metric(model, latency, worker_resp.status_code, None)

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            status_code=worker_resp.status_code,
        )

    # --- Non-streaming ---
    try:
        resp = await http_client.post(target, json=body)
    except Exception as exc:
        # Failover
        fallback_url = pool.next()
        if fallback_url and fallback_url != worker_url:
            try:
                resp = await http_client.post(
                    f"{fallback_url}/v1/chat/completions", json=body
                )
            except Exception:
                return JSONResponse(status_code=502, content={"error": f"All workers failed: {exc}"})
        else:
            return JSONResponse(status_code=502, content={"error": f"Worker unreachable: {exc}"})

    latency = (time.monotonic() - t0) * 1000
    resp_json = resp.json()
    tokens = None
    if "usage" in resp_json:
        tokens = resp_json["usage"].get("total_tokens")
    await _log_metric(model, latency, resp.status_code, tokens)

    return JSONResponse(status_code=resp.status_code, content=resp_json)


@app.get("/metrics")
async def metrics():
    try:
        raw = await redis_client.lrange("ssw:metrics", 0, 999)
    except Exception:
        return {"error": "Redis unavailable"}

    entries = [json.loads(r) for r in raw]

    # Aggregate by model
    stats: dict[str, dict] = {}
    for e in entries:
        m = e["model"]
        if m not in stats:
            stats[m] = {"count": 0, "errors": 0, "total_latency": 0.0, "total_tokens": 0}
        stats[m]["count"] += 1
        if e["status"] >= 400:
            stats[m]["errors"] += 1
        stats[m]["total_latency"] += e["latency_ms"]
        if e["tokens"]:
            stats[m]["total_tokens"] += e["tokens"]

    summary = {}
    for m, s in stats.items():
        summary[m] = {
            "request_count": s["count"],
            "error_count": s["errors"],
            "avg_latency_ms": round(s["total_latency"] / s["count"], 1) if s["count"] else 0,
            "total_tokens": s["total_tokens"],
        }

    # Active requests per worker (based on health status)
    workers = {}
    for name, pool in pools.items():
        for url in pool.all_urls():
            workers[url] = "healthy" if url in pool.healthy else "unhealthy"

    return {"models": summary, "workers": workers, "recent_entries": len(entries)}
