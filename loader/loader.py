"""Loader service — starts/stops vLLM containers via Docker SDK."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import docker
import httpx
import yaml
from fastapi import FastAPI
from fastapi.responses import JSONResponse

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GATEWAY_URL = os.environ.get("LOADER_GATEWAY_URL", "http://gateway:8000")
MODELS_DIR = os.environ.get("LOADER_MODELS_DIR", "/models")
MODELS_YAML = os.environ.get("LOADER_MODELS_YAML_PATH", "/app/models.yaml")
DOCKER_NETWORK = os.environ.get("LOADER_DOCKER_NETWORK", "sswai_default")
VLLM_IMAGE = os.environ.get("LOADER_VLLM_IMAGE", "vllm/vllm-openai:latest")
PORT_START = int(os.environ.get("LOADER_PORT_START", "9000"))
PORT_END = int(os.environ.get("LOADER_PORT_END", "9099"))
A40_VRAM_MB = 46068  # NVIDIA A40 total VRAM
NUM_GPUS = int(os.environ.get("LOADER_NUM_GPUS", "2"))
HEALTH_POLL_INTERVAL = 5
HEALTH_TIMEOUT = 180

VLLM_SUPPORTED_TYPES = {"qwen2", "qwen3", "llama", "mistral", "gemma", "gemma2", "phi3"}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class WorkerStatus(str, Enum):
    LOADING = "loading"
    READY = "ready"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class WorkerState:
    model_name: str  # always canonical short name
    container_id: str
    container_name: str
    gpu_id: int
    port: int
    url: str
    vram_mb: int
    status: WorkerStatus = WorkerStatus.LOADING
    last_request: float = 0.0
    pinned: bool = False
    error: str | None = None


@dataclass
class GPUState:
    gpu_id: int
    total_mb: int = A40_VRAM_MB
    workers: list[str] = field(default_factory=list)
    reserved_mb: int = 0  # VRAM claimed by in-progress loads

    @property
    def used_mb(self) -> int:
        return self.reserved_mb + sum(
            _loader_state.workers[name].vram_mb
            for name in self.workers
            if name in _loader_state.workers
        )

    @property
    def free_mb(self) -> int:
        return self.total_mb - self.used_mb


@dataclass
class ModelConfig:
    model_id: str
    name: str
    path: str
    architecture: str
    quant_method: str
    vram_mb: int
    kv_reserve_mb: int
    tool_call_parser: str
    thinking_suppression: bool
    dtype: str
    quantization: str | None
    max_model_len: int
    pinned: bool
    aliases: list[str]


class LoaderState:
    def __init__(self):
        self.workers: dict[str, WorkerState] = {}
        self.gpus: dict[int, GPUState] = {i: GPUState(gpu_id=i) for i in range(NUM_GPUS)}
        self.used_ports: set[int] = set()
        self.model_configs: dict[str, ModelConfig] = {}
        self._lock = asyncio.Lock()

    def allocate_port(self) -> int | None:
        for p in range(PORT_START, PORT_END + 1):
            if p not in self.used_ports:
                self.used_ports.add(p)
                return p
        return None

    def release_port(self, port: int):
        self.used_ports.discard(port)

    def resolve_model(self, name: str) -> ModelConfig | None:
        """Resolve any name/alias to canonical ModelConfig."""
        return self.model_configs.get(name)

    def is_loaded(self, mc: ModelConfig) -> bool:
        """Check if a model (by canonical name) is already loaded or loading."""
        ws = self.workers.get(mc.name)
        return ws is not None and ws.status in (WorkerStatus.READY, WorkerStatus.LOADING)

    def find_gpu(self, vram_needed: int) -> int | None:
        best_gpu = None
        best_free = -1
        for gpu_id, gpu in self.gpus.items():
            free = gpu.free_mb
            if free >= vram_needed and free > best_free:
                best_gpu = gpu_id
                best_free = free
        return best_gpu

    def find_eviction_candidates(self, gpu_id: int, need_mb: int) -> list[str]:
        gpu = self.gpus[gpu_id]
        evictable = []
        for name in gpu.workers:
            w = self.workers.get(name)
            if w and not w.pinned and w.status == WorkerStatus.READY:
                evictable.append(w)
        evictable.sort(key=lambda w: w.last_request)
        candidates = []
        freed = 0
        for w in evictable:
            candidates.append(w.model_name)
            freed += w.vram_mb
            if freed >= need_mb:
                break
        return candidates if freed >= need_mb else []


_loader_state = LoaderState()
_docker_client: docker.DockerClient | None = None


# ---------------------------------------------------------------------------
# Model config loading
# ---------------------------------------------------------------------------
def _load_model_configs():
    yaml_overrides = {}
    if os.path.isfile(MODELS_YAML):
        with open(MODELS_YAML) as f:
            data = yaml.safe_load(f) or {}
        yaml_overrides = {m["model_id"]: m for m in data.get("models", []) if "model_id" in m}

    base = Path(MODELS_DIR)
    if not base.is_dir():
        log.warning("Models dir %s not found", MODELS_DIR)
        return

    skip_prefixes = ("models--", "huggingface", "whisper", "crv_")

    def scan(model_dir: Path, model_id: str):
        config_path = model_dir / "config.json"
        if not config_path.is_file():
            return
        with open(config_path) as f:
            cfg = json.load(f)
        model_type = cfg.get("model_type", "")
        if model_type not in VLLM_SUPPORTED_TYPES:
            return
        quant = cfg.get("quantization_config", {}).get("quant_method", "none")
        ov = yaml_overrides.get(model_id, {})

        total_bytes = sum(
            (model_dir / f).stat().st_size
            for f in os.listdir(model_dir)
            if f.endswith((".safetensors", ".bin"))
        )
        disk_mb = total_bytes / (1024 * 1024)
        est_vram = int(disk_mb * (1.1 if quant in ("awq", "gptq", "awq_marlin") else 1.05))
        name = model_id.split("/")[-1].lower()

        mc = ModelConfig(
            model_id=model_id,
            name=name,
            path=str(model_dir),
            architecture=model_type,
            quant_method=quant,
            vram_mb=ov.get("vram_mb", est_vram),
            kv_reserve_mb=ov.get("kv_reserve_mb", 4096),
            tool_call_parser=ov.get("tool_call_parser", _infer_parser(model_type)),
            thinking_suppression=ov.get("thinking_suppression", model_type == "qwen3"),
            dtype=ov.get("dtype", "half" if quant in ("awq", "gptq") else "auto"),
            quantization=ov.get("quantization", _infer_quant_flag(quant)),
            max_model_len=ov.get("max_model_len", 32768),
            pinned=ov.get("pinned", False),
            aliases=ov.get("aliases", []),
        )
        _loader_state.model_configs[name] = mc
        _loader_state.model_configs[model_id] = mc
        for alias in mc.aliases:
            _loader_state.model_configs[alias] = mc

    for entry in sorted(base.iterdir()):
        if not entry.is_dir():
            continue
        if any(entry.name.startswith(p) for p in skip_prefixes):
            continue
        if (entry / "config.json").is_file():
            scan(entry, entry.name)
        else:
            for sub in sorted(entry.iterdir()):
                if sub.is_dir() and (sub / "config.json").is_file():
                    scan(sub, f"{entry.name}/{sub.name}")

    log.info("Loaded %d model configs", len(set(id(v) for v in _loader_state.model_configs.values())))


def _infer_parser(model_type: str) -> str:
    if model_type in ("qwen2", "qwen3"):
        return "hermes"
    if model_type == "llama":
        return "llama3_json"
    if model_type == "mistral":
        return "mistral"
    return "hermes"


def _infer_quant_flag(quant: str) -> str | None:
    if quant == "awq":
        return "awq_marlin"
    if quant == "gptq":
        return "gptq_marlin"
    return None


# ---------------------------------------------------------------------------
# Docker operations
# ---------------------------------------------------------------------------
def _get_docker() -> docker.DockerClient:
    global _docker_client
    if _docker_client is None:
        _docker_client = docker.from_env()
    return _docker_client


def _spawn_vllm_container(mc: ModelConfig, gpu_id: int, port: int) -> str:
    client = _get_docker()
    container_name = f"ssw-{mc.name}-gpu{gpu_id}"

    # Compute GPU memory utilization fraction (I5)
    total_model_vram = mc.vram_mb + mc.kv_reserve_mb
    gpu_util = min(round(total_model_vram / A40_VRAM_MB, 2), 0.95)

    cmd = [
        f"--model=/models/{mc.model_id}",
        f"--served-model-name={mc.name}",
        "--port=8000",
        f"--gpu-memory-utilization={gpu_util}",
        f"--max-model-len={mc.max_model_len}",
        f"--tool-call-parser={mc.tool_call_parser}",
        "--enable-auto-tool-choice",
        f"--dtype={mc.dtype}",
    ]
    if mc.quantization:
        cmd.append(f"--quantization={mc.quantization}")

    try:
        old = client.containers.get(container_name)
        old.remove(force=True)
    except docker.errors.NotFound:
        pass

    container = client.containers.run(
        VLLM_IMAGE,
        command=cmd,
        name=container_name,
        detach=True,
        network=DOCKER_NETWORK,
        volumes={MODELS_DIR: {"bind": "/models", "mode": "ro"}},
        ports={"8000/tcp": ("127.0.0.1", port)},
        device_requests=[
            docker.types.DeviceRequest(
                device_ids=[str(gpu_id)],
                capabilities=[["gpu"]],
            )
        ],
        restart_policy={"Name": "unless-stopped"},
    )

    log.info(
        "Spawned container %s (id=%s) on GPU %d, port %d for %s (gpu_util=%.2f)",
        container_name, container.id[:12], gpu_id, port, mc.name, gpu_util,
    )
    return container.id


async def _wait_for_health(url: str, timeout: int = HEALTH_TIMEOUT) -> bool:
    deadline = time.monotonic() + timeout
    async with httpx.AsyncClient() as client:
        while time.monotonic() < deadline:
            try:
                resp = await client.get(f"{url}/health", timeout=5.0)
                if resp.status_code == 200:
                    return True
            except Exception:
                pass
            await asyncio.sleep(HEALTH_POLL_INTERVAL)
    return False


async def _register_with_gateway(model_name: str, worker_url: str):
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{GATEWAY_URL}/internal/workers/register",
            json={"model_name": model_name, "url": worker_url},
            timeout=10.0,
        )
        resp.raise_for_status()
        log.info("Registered %s at %s with gateway", model_name, worker_url)


async def _unregister_from_gateway(model_name: str, worker_url: str):
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{GATEWAY_URL}/internal/workers/unregister",
            json={"model_name": model_name, "url": worker_url},
            timeout=10.0,
        )
        resp.raise_for_status()
        log.info("Unregistered %s at %s from gateway", model_name, worker_url)


async def _stop_container(container_id: str):
    client = _get_docker()
    try:
        container = client.containers.get(container_id)
        container.stop(timeout=30)
        container.remove()
    except docker.errors.NotFound:
        pass
    except Exception as e:
        log.warning("Error stopping container %s: %s", container_id[:12], e)


# ---------------------------------------------------------------------------
# Core load/unload
# ---------------------------------------------------------------------------
async def _do_load(model_name: str, gpu_id: int | None = None) -> WorkerState:
    async with _loader_state._lock:
        # C4: resolve alias to canonical ModelConfig
        mc = _loader_state.resolve_model(model_name)
        if mc is None:
            raise ValueError(f"Unknown model: {model_name!r}")

        # Check if already loaded under canonical name
        if _loader_state.is_loaded(mc):
            ws = _loader_state.workers[mc.name]
            raise ValueError(f"Model {mc.name!r} already {ws.status.value}")

        total_vram = mc.vram_mb + mc.kv_reserve_mb

        # Find GPU
        if gpu_id is None:
            gpu_id = _loader_state.find_gpu(total_vram)

        evict_names = []
        if gpu_id is None:
            # Try eviction on each GPU
            for gid in range(NUM_GPUS):
                gpu = _loader_state.gpus[gid]
                candidates = _loader_state.find_eviction_candidates(gid, total_vram - gpu.free_mb)
                if candidates:
                    gpu_id = gid
                    evict_names = candidates
                    break

        if gpu_id is None:
            raise RuntimeError(
                f"No GPU has {total_vram}MB free and no evictable models found"
            )

        # If we have a GPU but still need eviction
        if not evict_names:
            gpu = _loader_state.gpus[gpu_id]
            if gpu.free_mb < total_vram:
                evict_names = _loader_state.find_eviction_candidates(
                    gpu_id, total_vram - gpu.free_mb
                )
                if not evict_names:
                    raise RuntimeError(
                        f"GPU {gpu_id} needs {total_vram}MB but only {gpu.free_mb}MB free "
                        f"and no evictable models"
                    )

        port = _loader_state.allocate_port()
        if port is None:
            raise RuntimeError("No available ports in dynamic range")

        # C2: Reserve VRAM on GPU while lock is held, before we release it
        _loader_state.gpus[gpu_id].reserved_mb += total_vram

    # Evict if needed (outside lock, but VRAM is reserved)
    for ename in evict_names:
        await _do_unload(ename)

    # Spawn container
    try:
        container_id = _spawn_vllm_container(mc, gpu_id, port)
    except Exception as e:
        # Release reservation on failure
        async with _loader_state._lock:
            _loader_state.gpus[gpu_id].reserved_mb -= total_vram
            _loader_state.release_port(port)
        raise RuntimeError(f"Failed to spawn container: {e}")

    container_name = f"ssw-{mc.name}-gpu{gpu_id}"
    worker_url = f"http://{container_name}:8000"

    ws = WorkerState(
        model_name=mc.name,
        container_id=container_id,
        container_name=container_name,
        gpu_id=gpu_id,
        port=port,
        url=worker_url,
        vram_mb=total_vram,
        pinned=mc.pinned,
    )

    async with _loader_state._lock:
        _loader_state.workers[mc.name] = ws
        _loader_state.gpus[gpu_id].workers.append(mc.name)
        # Clear reservation — VRAM is now tracked via workers list
        _loader_state.gpus[gpu_id].reserved_mb -= total_vram

    # Wait for health
    healthy = await _wait_for_health(worker_url)
    if not healthy:
        ws.status = WorkerStatus.ERROR
        ws.error = "Health check timeout"
        log.error("Model %s failed health check after %ds", mc.name, HEALTH_TIMEOUT)
        await _stop_container(container_id)
        async with _loader_state._lock:
            if mc.name in _loader_state.gpus[gpu_id].workers:
                _loader_state.gpus[gpu_id].workers.remove(mc.name)
            if mc.name in _loader_state.workers:
                del _loader_state.workers[mc.name]
            _loader_state.release_port(port)
        raise RuntimeError(f"Model {mc.name} failed to start within {HEALTH_TIMEOUT}s")

    ws.status = WorkerStatus.READY
    ws.last_request = time.time()

    # Register with gateway under canonical name + aliases
    await _register_with_gateway(mc.name, worker_url)
    for alias in mc.aliases:
        await _register_with_gateway(alias, worker_url)

    return ws


async def _do_unload(model_name: str) -> None:
    async with _loader_state._lock:
        # C4: resolve alias to canonical name
        mc = _loader_state.resolve_model(model_name)
        canonical = mc.name if mc else model_name
        ws = _loader_state.workers.get(canonical)
        if ws is None:
            raise ValueError(f"Model {model_name!r} is not loaded")
        if ws.pinned:
            raise ValueError(f"Model {model_name!r} is pinned and cannot be unloaded")
        ws.status = WorkerStatus.STOPPING

    mc = _loader_state.model_configs.get(canonical)

    try:
        await _unregister_from_gateway(ws.model_name, ws.url)
        if mc:
            for alias in mc.aliases:
                await _unregister_from_gateway(alias, ws.url)
    except Exception as e:
        log.warning("Failed to unregister %s from gateway: %s", canonical, e)

    await _stop_container(ws.container_id)

    async with _loader_state._lock:
        gpu = _loader_state.gpus[ws.gpu_id]
        if ws.model_name in gpu.workers:
            gpu.workers.remove(ws.model_name)
        _loader_state.release_port(ws.port)
        if canonical in _loader_state.workers:
            del _loader_state.workers[canonical]

    log.info("Unloaded %s from GPU %d", canonical, ws.gpu_id)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model_configs()
    log.info("Loader ready with %d model configs",
             len(set(id(v) for v in _loader_state.model_configs.values())))
    yield


app = FastAPI(title="SSW Model Loader", lifespan=lifespan)


@app.post("/load")
async def load_model(request_body: dict):
    model_name = request_body.get("model_name")
    gpu_id = request_body.get("gpu_id")
    if not model_name:
        return JSONResponse(status_code=400, content={"error": "model_name required"})

    try:
        ws = await _do_load(model_name, gpu_id)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except RuntimeError as e:
        return JSONResponse(status_code=503, content={"error": str(e)})

    return {
        "status": ws.status.value,
        "model": ws.model_name,
        "gpu_id": ws.gpu_id,
        "port": ws.port,
        "url": ws.url,
        "vram_mb": ws.vram_mb,
    }


@app.post("/unload")
async def unload_model(request_body: dict):
    model_name = request_body.get("model_name")
    if not model_name:
        return JSONResponse(status_code=400, content={"error": "model_name required"})

    try:
        await _do_unload(model_name)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    return {"status": "unloaded", "model": model_name}


@app.get("/status")
async def status():
    workers = {}
    for name, ws in _loader_state.workers.items():
        workers[name] = {
            "container_id": ws.container_id[:12],
            "container_name": ws.container_name,
            "gpu_id": ws.gpu_id,
            "port": ws.port,
            "url": ws.url,
            "vram_mb": ws.vram_mb,
            "status": ws.status.value,
            "pinned": ws.pinned,
            "last_request": ws.last_request,
            "error": ws.error,
        }

    gpus = {}
    for gid, gpu in _loader_state.gpus.items():
        gpus[str(gid)] = {
            "total_mb": gpu.total_mb,
            "used_mb": gpu.used_mb,
            "free_mb": gpu.free_mb,
            "reserved_mb": gpu.reserved_mb,
            "workers": list(gpu.workers),
        }

    available = []
    loaded_names = set(_loader_state.workers.keys())
    seen = set()
    for name, mc in _loader_state.model_configs.items():
        if mc.model_id in seen:
            continue
        seen.add(mc.model_id)
        available.append({
            "name": mc.name,
            "model_id": mc.model_id,
            "vram_mb": mc.vram_mb,
            "kv_reserve_mb": mc.kv_reserve_mb,
            "total_vram_mb": mc.vram_mb + mc.kv_reserve_mb,
            "pinned": mc.pinned,
            "aliases": mc.aliases,
            "loaded": mc.name in loaded_names,
        })

    return {
        "workers": workers,
        "gpus": gpus,
        "available_models": available,
        "ports_in_use": sorted(_loader_state.used_ports),
    }


@app.patch("/touch")
async def touch_model(request_body: dict):
    model_name = request_body.get("model_name")
    if not model_name:
        return JSONResponse(status_code=400, content={"error": "model_name required"})
    # Resolve alias
    mc = _loader_state.resolve_model(model_name)
    canonical = mc.name if mc else model_name
    ws = _loader_state.workers.get(canonical)
    if ws is None:
        return JSONResponse(status_code=404, content={"error": f"Model {model_name!r} not loaded"})
    ws.last_request = time.time()
    return {"status": "touched", "model": canonical, "last_request": ws.last_request}


@app.get("/health")
async def health():
    return {"status": "ok"}
