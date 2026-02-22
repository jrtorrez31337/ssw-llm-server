import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager

_GATEWAY_CREATED = int(time.time())

import httpx
import redis.asyncio as redis
from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse

from config import PoolManager, Settings, settings
from request_queue import RequestQueue
from registry import ModelInfo, load_registry

log = logging.getLogger(__name__)

LOADER_URL = os.environ.get("GATEWAY_LOADER_URL", "http://loader:8001")


def _oai_error(status: int, message: str, error_type: str = "invalid_request_error",
               code: str | None = None) -> JSONResponse:
    """Return an OpenAI-compatible error response."""
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": error_type, "param": None, "code": code}},
    )

# ---------------------------------------------------------------------------
# Globals (initialized in lifespan)
# ---------------------------------------------------------------------------
pool_mgr = PoolManager()
model_registry: dict[str, ModelInfo] = {}
http_client: httpx.AsyncClient = None  # type: ignore[assignment]
redis_client: redis.Redis = None  # type: ignore[assignment]
request_queue: RequestQueue = None  # type: ignore[assignment]
_health_task: asyncio.Task | None = None


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
def _register_static_workers():
    """Register workers from legacy env vars (backward compat)."""
    for url in settings.heavy_workers.split(","):
        url = url.strip()
        if url:
            pool_mgr.register_worker("heavy", url)
    for url in settings.light_workers.split(","):
        url = url.strip()
        if url:
            pool_mgr.register_worker("light", url)
    log.info(
        "Static workers registered: heavy=%d, light=%d",
        pool_mgr.get_pool("heavy").worker_count() if pool_mgr.get_pool("heavy") else 0,
        pool_mgr.get_pool("light").worker_count() if pool_mgr.get_pool("light") else 0,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_registry, http_client, redis_client, request_queue, _health_task

    model_registry = load_registry(settings.models_dir, settings.models_yaml_path)
    _register_static_workers()

    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(
            settings.worker_timeout, connect=settings.worker_connect_timeout
        ),
        limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
    )
    redis_client = redis.from_url(settings.redis_url, decode_responses=True)
    request_queue = RequestQueue(redis_client)
    _health_task = asyncio.create_task(_health_loop())

    yield

    _health_task.cancel()
    await http_client.aclose()
    await redis_client.aclose()


app = FastAPI(title="SSW AI Gateway", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Health-check background loop
# ---------------------------------------------------------------------------
async def _check_worker(pool_name: str, url: str):
    pool = pool_mgr.get_pool(pool_name)
    if pool is None:
        return
    try:
        resp = await http_client.get(f"{url}/health", timeout=5.0)
        if resp.status_code == 200:
            pool.mark_healthy(url)
        else:
            pool.mark_unhealthy(url)
    except Exception:
        pool.mark_unhealthy(url)

    # Scrape vLLM queue depth
    try:
        resp = await http_client.get(f"{url}/metrics", timeout=5.0)
        if resp.status_code == 200:
            depth = _parse_queue_depth(resp.text)
            pool.set_queue_depth(url, depth)
    except Exception:
        pass


def _parse_queue_depth(metrics_text: str) -> int:
    """Extract running + waiting request count from Prometheus metrics."""
    running = 0
    waiting = 0
    for line in metrics_text.splitlines():
        if line.startswith("vllm:num_requests_running{"):
            running = int(float(line.rsplit(" ", 1)[1]))
        elif line.startswith("vllm:num_requests_waiting{"):
            waiting = int(float(line.rsplit(" ", 1)[1]))
    return running + waiting


async def _health_loop():
    while True:
        tasks = []
        for name, pool in pool_mgr.all_pools().items():
            for url in pool.all_urls():
                tasks.append(_check_worker(name, url))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.sleep(settings.health_check_interval)


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------
def _resolve_model(model: str) -> tuple[str, ModelInfo | None]:
    """Resolve model name/alias to (pool_name, ModelInfo)."""
    info = model_registry.get(model)
    if info is None:
        return model, None

    # Check pools in priority order: requested name, canonical name, model_id, aliases
    for candidate in [model, info.name, info.model_id] + info.aliases:
        pool = pool_mgr.get_pool(candidate)
        if pool and pool.worker_count() > 0:
            return candidate, info

    return info.name, info


class _Saturated(Exception):
    """Raised when all workers in a pool exceed max queue depth."""


def _select_worker(model: str) -> tuple[str, str, ModelInfo | None]:
    """Pick pool by model name and return (pool_name, worker_url, ModelInfo)."""
    pool_name, info = _resolve_model(model)
    pool = pool_mgr.get_pool(pool_name)

    if pool is None or pool.worker_count() == 0:
        if info:
            raise LookupError(
                f"Model {model!r} is available but not loaded. "
                f"Use the loader API to start it."
            )
        raise ValueError(f"Unknown model: {model!r}")

    url = pool.next()
    if url is None:
        raise RuntimeError(f"No healthy workers for model {model!r}")

    # Backpressure: reject if every healthy worker exceeds max queue depth
    if settings.max_queue_depth > 0:
        least_loaded = pool.min_queue_worker()
        if least_loaded and pool.queue_depth(least_loaded) >= settings.max_queue_depth:
            raise _Saturated(
                f"All workers for model {model!r} are at capacity "
                f"(queue depth >= {settings.max_queue_depth})"
            )
        # Route to least-loaded worker instead of round-robin when under pressure
        if least_loaded and pool.queue_depth(url) > pool.queue_depth(least_loaded):
            url = least_loaded

    return pool_name, url, info


def _worker_body(body: dict, pool_name: str) -> dict:
    """Return request body normalized for worker-served model names."""
    worker_body = dict(body)
    worker_body["model"] = pool_name
    return worker_body


def _set_response_model(resp_body: object, requested_model: str) -> None:
    """Normalize worker response model field back to caller-facing model name."""
    if isinstance(resp_body, dict) and "model" in resp_body:
        resp_body["model"] = requested_model


def _normalize_sse_line(line: str, requested_model: str) -> str:
    """Rewrite model field in SSE data lines when payload is JSON."""
    if not line.startswith("data:"):
        return line

    payload = line[len("data:"):].lstrip()
    if not payload or payload == "[DONE]":
        return line

    try:
        obj = json.loads(payload)
    except Exception:
        return line

    _set_response_model(obj, requested_model)
    return f"data: {json.dumps(obj, separators=(',', ':'))}"


async def _log_metric(model: str, latency_ms: float, status: int, tokens: int | None):
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
        pass


# ---------------------------------------------------------------------------
# Auto-load + queue
# ---------------------------------------------------------------------------
async def _trigger_loader(model: str):
    """Ask the loader to start a model (fire-and-forget)."""
    try:
        resp = await http_client.post(
            f"{LOADER_URL}/load",
            json={"model_name": model},
            timeout=10.0,
        )
        if resp.status_code == 200:
            log.info("Loader accepted load request for %s", model)
        else:
            log.warning("Loader rejected load for %s: %s", model, resp.text)
    except Exception as e:
        log.warning("Failed to reach loader for %s: %s", model, e)


async def _handle_unloaded_model(model: str, body: dict, request: Request):
    """Handle a request for an unloaded model: trigger loader, optionally queue."""
    wait_header = request.headers.get("x-ssw-wait", "").lower()
    want_wait = wait_header in ("true", "1", "yes")

    # Trigger the loader in the background
    asyncio.create_task(_trigger_loader(model))

    if not want_wait:
        return JSONResponse(
            status_code=202,
            content={
                "status": "loading",
                "model": model,
                "message": f"Model {model!r} is being loaded. Retry shortly.",
            },
            headers={"Retry-After": "60"},
        )

    # Client wants to wait — subscribe BEFORE enqueuing to avoid pub/sub race (C3)
    request_id, response = await request_queue.enqueue_and_wait(model, body, timeout=180)

    if response is None:
        return _oai_error(504, f"Timed out waiting for model {model!r} to load",
                          error_type="server_error", code="timeout")

    return JSONResponse(
        status_code=response.get("status_code", 200),
        content=response.get("body", {}),
    )


async def _drain_and_execute(model: str):
    """Drain queued requests for a model and execute them."""
    entries = await request_queue.drain_queue(model)
    for entry in entries:
        asyncio.create_task(_execute_queued_request(entry))


async def _execute_queued_request(entry: dict):
    """Execute a single queued request and publish the response."""
    request_id = entry["request_id"]
    model = entry["model"]
    body = entry["body"]

    # Skip stale entries (I4)
    age = time.time() - entry.get("ts", 0)
    if age > 180:
        log.info("Skipping stale queued request %s (age=%.0fs)", request_id, age)
        await request_queue.publish_response(request_id, {
            "status_code": 504,
            "body": {"error": {"message": "Request expired while waiting for model to load",
                               "type": "server_error", "param": None, "code": "timeout"}},
        })
        return

    try:
        pool_name, worker_url, model_info = _select_worker(model)
    except Exception as e:
        await request_queue.publish_response(request_id, {
            "status_code": 503,
            "body": {"error": {"message": str(e), "type": "server_error", "param": None, "code": None}},
        })
        return

    if model_info and model_info.thinking_suppression:
        body.setdefault("chat_template_kwargs", {})["enable_thinking"] = False

    target = f"{worker_url}/v1/chat/completions"
    worker_body = _worker_body(body, pool_name)
    worker_body["stream"] = False

    try:
        resp = await http_client.post(target, json=worker_body)
        try:
            resp_body = resp.json()
        except Exception:
            resp_body = {"error": {"message": resp.text[:500], "type": "server_error", "param": None, "code": None}}
        _set_response_model(resp_body, model)
        await request_queue.publish_response(request_id, {
            "status_code": resp.status_code,
            "body": resp_body,
        })
    except Exception as e:
        await request_queue.publish_response(request_id, {
            "status_code": 502,
            "body": {"error": {"message": f"Worker failed: {e}", "type": "server_error", "param": None, "code": None}},
        })


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    worker_status = {}
    for name, pool in pool_mgr.all_pools().items():
        for url, healthy in pool.health_snapshot().items():
            worker_status[url] = "healthy" if healthy else "unhealthy"
    healthy_count = sum(1 for v in worker_status.values() if v == "healthy")
    any_healthy = healthy_count > 0
    all_healthy = healthy_count == len(worker_status) and healthy_count > 0
    status = "ok" if all_healthy else "degraded" if any_healthy else "down"
    return JSONResponse(
        status_code=200 if any_healthy else 503,
        content={"status": status, "workers": worker_status},
    )


def _build_model_entry(info: ModelInfo, public_id: str | None = None) -> dict:
    """Build an OpenAI-compatible model object with SSW extensions."""
    active_pools = pool_mgr.all_pools()
    workers = 0
    healthy = 0
    for check_name in [info.name, info.model_id] + info.aliases:
        pool = active_pools.get(check_name)
        if pool and pool.worker_count() > 0:
            workers += pool.worker_count()
            healthy += pool.healthy_count()

    entry = {
        "id": public_id or info.name,
        "object": "model",
        "created": _GATEWAY_CREATED,
        "owned_by": "ssw",
        # SSW extensions
        "model_id": info.model_id,
        "status": "loaded" if workers > 0 else "available",
        "workers": workers,
        "healthy_workers": healthy,
        "pinned": info.pinned,
        "vram_mb": info.vram_mb,
        "aliases": info.aliases,
        "architecture": info.architecture,
        "quant_method": info.quant_method,
    }
    return entry


@app.get("/v1/models")
async def list_models():
    seen_model_ids = set()
    emitted_ids: set[str] = set()
    data = []
    for name, info in model_registry.items():
        if info.model_id in seen_model_ids:
            continue
        seen_model_ids.add(info.model_id)

        for public_id in [info.name] + info.aliases:
            if public_id in emitted_ids:
                continue
            emitted_ids.add(public_id)
            data.append(_build_model_entry(info, public_id=public_id))
    return {"object": "list", "data": data}


@app.get("/v1/models/status")
async def models_status():
    seen = set()
    loaded = []
    available = []
    for name, info in model_registry.items():
        if info.model_id in seen:
            continue
        seen.add(info.model_id)
        entry = _build_model_entry(info)
        if entry["status"] == "loaded":
            loaded.append(entry)
        else:
            available.append(entry)
    return {"loaded": loaded, "available": available}


@app.get("/v1/models/{model_id:path}")
async def retrieve_model(model_id: str):
    info = model_registry.get(model_id)
    if info is None:
        return _oai_error(404, f"The model '{model_id}' does not exist",
                          code="model_not_found")
    public_id = model_id if model_id in info.aliases else info.name
    return _build_model_entry(info, public_id=public_id)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model = body.get("model", "heavy")
    is_stream = body.get("stream", False)

    # Select worker — with queue support for unloaded models
    try:
        pool_name, worker_url, model_info = _select_worker(model)
    except LookupError:
        return await _handle_unloaded_model(model, body, request)
    except ValueError as e:
        return _oai_error(404, str(e), code="model_not_found")
    except _Saturated as e:
        return JSONResponse(
            status_code=429,
            content={"error": {"message": str(e), "type": "rate_limit_error", "param": None, "code": "engine_overloaded"}},
            headers={"Retry-After": str(settings.health_check_interval)},
        )
    except RuntimeError as e:
        return _oai_error(503, str(e), error_type="server_error", code="engine_overloaded")

    # Thinking suppression based on registry
    if model_info and model_info.thinking_suppression:
        body.setdefault("chat_template_kwargs", {})["enable_thinking"] = False

    pool = pool_mgr.get_pool(pool_name)
    target = f"{worker_url}/v1/chat/completions"
    worker_body = _worker_body(body, pool_name)
    t0 = time.monotonic()

    # --- Streaming ---
    if is_stream:
        try:
            worker_req = http_client.build_request("POST", target, json=worker_body)
            worker_resp = await http_client.send(worker_req, stream=True)
        except Exception as exc:
            fallback_url = pool.next() if pool else None
            if fallback_url and fallback_url != worker_url:
                try:
                    worker_req = http_client.build_request(
                        "POST", f"{fallback_url}/v1/chat/completions", json=worker_body
                    )
                    worker_resp = await http_client.send(worker_req, stream=True)
                except Exception:
                    return _oai_error(502, f"All workers failed: {exc}", error_type="server_error")
            else:
                return _oai_error(502, f"Worker unreachable: {exc}", error_type="server_error")

        async def stream_generator():
            try:
                async for line in worker_resp.aiter_lines():
                    yield (_normalize_sse_line(line, model) + "\n").encode("utf-8")
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
        resp = await http_client.post(target, json=worker_body)
    except Exception as exc:
        fallback_url = pool.next() if pool else None
        if fallback_url and fallback_url != worker_url:
            try:
                resp = await http_client.post(
                    f"{fallback_url}/v1/chat/completions", json=worker_body
                )
            except Exception:
                return _oai_error(502, f"All workers failed: {exc}", error_type="server_error")
        else:
            return _oai_error(502, f"Worker unreachable: {exc}", error_type="server_error")

    latency = (time.monotonic() - t0) * 1000
    try:
        resp_json = resp.json()
    except Exception:
        resp_json = {"error": {"message": resp.text[:500], "type": "server_error", "param": None, "code": None}}
    _set_response_model(resp_json, model)
    tokens = None
    if "usage" in resp_json:
        tokens = resp_json["usage"].get("total_tokens")
    await _log_metric(model, latency, resp.status_code, tokens)

    return JSONResponse(status_code=resp.status_code, content=resp_json)


# ---------------------------------------------------------------------------
# Internal worker management API (called by loader)
# ---------------------------------------------------------------------------
@app.post("/internal/workers/register")
async def register_worker(request: Request):
    body = await request.json()
    model_name = body.get("model_name")
    url = body.get("url")
    if not model_name or not url:
        return _oai_error(400, "model_name and url required")
    pool = pool_mgr.register_worker(model_name, url)
    log.info("Worker registered: model=%s url=%s total=%d", model_name, url, pool.worker_count())

    # Drain any queued requests for this model (and aliases)
    asyncio.create_task(_drain_and_execute(model_name))
    info = model_registry.get(model_name)
    if info:
        for alias in info.aliases:
            if alias != model_name:
                asyncio.create_task(_drain_and_execute(alias))

    return {"status": "registered", "model": model_name, "url": url, "workers": pool.worker_count()}


@app.post("/internal/workers/unregister")
async def unregister_workers(request: Request):
    body = await request.json()
    model_name = body.get("model_name")
    url = body.get("url")
    if not model_name or not url:
        return _oai_error(400, "model_name and url required")
    ok = pool_mgr.unregister_worker(model_name, url)
    if not ok:
        return _oai_error(404, f"Pool {model_name!r} not found", code="not_found")
    log.info("Worker unregistered: model=%s url=%s", model_name, url)
    return {"status": "unregistered", "model": model_name, "url": url}


@app.get("/metrics")
async def metrics():
    try:
        raw = await redis_client.lrange("ssw:metrics", 0, 999)
    except Exception:
        return {"error": "Redis unavailable"}

    entries = [json.loads(r) for r in raw]

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

    workers = {}
    for name, pool in pool_mgr.all_pools().items():
        for url, healthy in pool.health_snapshot().items():
            workers[url] = "healthy" if healthy else "unhealthy"

    return {"models": summary, "workers": workers, "recent_entries": len(entries)}
