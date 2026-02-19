# SSW LLM Server

Open-source LLM inference platform built for **Social Space Wars** — a galaxy-scale MMO where NPC behavior is driven by large language models instead of scripted behavior trees.

**This repo is the server** — the inference backbone that serves models over an OpenAI-compatible API. The NPC agent logic (memory, doctrine, action selection, context engineering) lives client-side. This server doesn't know or care what the agents do; it just serves tokens fast.

## Why This Exists

The research below shaped the *client-side* agent design for SSW — not this server directly. We include it here for context on what this infrastructure is built to support: large-scale autonomous NPC populations that need reliable, high-throughput inference.

**[Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442)** — Park et al. (Stanford/Google, 2023). 25 AI agents in a virtual village developed realistic personalities, social connections, and multi-step plans autonomously. Established that memory + reflection mechanisms are load-bearing for believable NPC behavior.

**[Project Sid: Many-Agent Simulations Toward AI Civilization](https://arxiv.org/abs/2411.00114)** — Altera.AL, 2024. 1,000+ autonomous agents in Minecraft created emergent civilization — specialized roles, democratic governance, taxation, and organically spreading culture — all without instruction.

**[ChatDev: Communicative Agents for Software Development](https://arxiv.org/abs/2307.07924)** — Qian et al. (ACL 2024). Multi-agent role specialization where agents with different roles coordinate autonomously. Validates doctrine-based conditioning for task specialization.

**[Generative Exaggeration in LLM Social Agents](https://arxiv.org/abs/2507.00657)** — 2025. Agents fabricate emotional understanding beyond evidence. Architectural constraints — not training alone — are necessary to prevent misalignment.

**[Effective Context Engineering for AI Agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)** — Anthropic Applied AI, 2025. Strategically curating what enters the context window is the key lever for agent reliability. Directly informs how NPC clients structure their requests to this server.

### Client-Side Design Principles (Not Implemented Here)

These principles guide the NPC agents that *call* this server:

- **Emergence over scripting** — Complex NPC behavior arises from simple components (memory, doctrine, action selection), not behavior trees
- **Doctrine conditioning** — Faction/archetype preference weights shape behavior without explicit rules
- **Active inference-lite** — NPCs select actions by minimizing prediction error: viability + preferences + information gain
- **Memory with decay** — Structured memory where older events lose salience, keeping context bounded
- **Context engineering** — Agents curate what enters the context window to maximize coherence within token budgets
- **Architectural safety** — Agent preferences are constrained to in-world outcomes; self-preservation cannot become a terminal goal

## Architecture

```
NPC Clients (OpenAI SDK compatible)
  │
  POST /v1/chat/completions  (model: "heavy", "light", or any model name)
  │
  ▼
FastAPI Gateway (:8000)  ─── 0.0.0.0, accepts remote clients
  │
  ├─ Model Registry ── scans /data/models, discovers all vLLM-compatible models
  │                     merges with models.yaml overrides (VRAM, aliases, pinned)
  │
  ├─ _resolve_model() ── alias lookup, pool matching
  │   ├─ "heavy" → alias → qwen2.5-14b-instruct-awq pool
  │   ├─ "light" → alias → qwen3-4b-awq pool
  │   └─ "qwen3-8b-awq" → direct name → pool (if loaded)
  │
  ├─ If model loaded → round-robin across healthy workers → proxy to vLLM
  │
  ├─ If model NOT loaded:
  │   ├─ No X-SSW-Wait header → 202 + Retry-After: 60
  │   └─ X-SSW-Wait: true → trigger loader, enqueue, block up to 180s
  │
  ├─ Failover → retry next healthy worker on connection failure
  │
  └─ Redis ← request metrics (latency, tokens, errors)

Loader Service (:8001)  ─── 127.0.0.1 only
  │
  ├─ POST /load  → find GPU, spawn vLLM container, health-check, register with gateway
  ├─ POST /unload → unregister from gateway, stop container, free VRAM
  ├─ GET /status → GPU utilization, all workers, available models
  │
  ├─ GPU Allocator → picks GPU with most free VRAM
  ├─ LRU Eviction → evicts oldest non-pinned model when VRAM is needed
  └─ Docker SDK → manages vLLM container lifecycle
```

### Available Models

The registry auto-discovers all vLLM-compatible models on disk. Currently 9 models:

| Model | Architecture | Quantization | Weight VRAM | Notes |
|-------|-------------|-------------|-------------|-------|
| `Qwen/Qwen2.5-14B-Instruct-AWQ` | qwen2 | AWQ | ~10.4 GB | **Pinned as `heavy`** — complex reasoning, diplomacy, quest generation |
| `Qwen/Qwen3-4B-AWQ` | qwen3 | AWQ | ~2.8 GB | **Pinned as `light`** — ambient NPC ticks, fast dialogue |
| `Qwen/Qwen3-8B-AWQ` | qwen3 | AWQ | ~6.4 GB | Mid-tier option for balance of speed + reasoning |
| `Qwen/Qwen3-14B-AWQ` | qwen3 | AWQ | ~10.4 GB | Latest Qwen3 14B, alternative to qwen2.5 heavy |
| `Qwen/Qwen3-4B-Instruct-2507-AWQ` | qwen3 | AWQ | ~2.8 GB | Updated 4B instruct variant |
| `meta-llama/Llama-3.1-8B-Instruct` | llama | BF16 | ~16 GB | Llama 3.1, llama3_json tool parser |
| `mistralai/Mistral-7B-Instruct-v0.2` | mistral | BF16 | ~14.8 GB | Mistral v0.2, mistral tool parser |
| `Orion-zhen/Qwen2.5-7B-Instruct-Uncensored` | qwen2 | BF16 | ~15.2 GB | Uncensored qwen2 variant |
| `thirdeyeai/DeepSeek-R1-Distill-Qwen-14B-uncensored` | qwen2 | BF16 | ~29.6 GB | DeepSeek R1 distillation |

All models support **tool/function calling** via their respective parsers (hermes for qwen, llama3_json for llama, mistral for mistral), configured automatically by the registry based on `model_type` from each model's `config.json`.

**Thinking suppression** is automatically applied to Qwen3 models — `enable_thinking: false` is injected to prevent `<think>` tag leakage in NPC dialogue.

### Hardware

- **2x NVIDIA A40 GPUs** (48 GB VRAM each, 96 GB total)
- Default deployment: each GPU runs 1x 14B worker (~10.4 GB) + 1x 4B worker (~2.8 GB)
- Remaining VRAM (~35 GB per GPU) available for KV cache and dynamically loaded models
- The loader service manages GPU placement and VRAM budgets when loading additional models

### Static vs Dynamic Workers

The platform supports two modes simultaneously:

**Static workers** are defined in `docker-compose.yml` and start with the stack. The default config starts 4 workers: `heavy-0` and `light-0` on GPU 0, `heavy-1` and `light-1` on GPU 1. These are the production workhorses — always available, pinned (never evicted).

**Dynamic workers** are started on-demand by the loader service. When a client requests a model that isn't loaded, the gateway triggers the loader, which spawns a vLLM container on the GPU with the most free VRAM. If VRAM is insufficient, the loader evicts the least-recently-used non-pinned model. Dynamic workers are registered with the gateway's pool manager and serve requests identically to static workers.

## Quick Start

```bash
# 1. Clone
git clone https://github.com/jrtorrez31337/ssw-llm-server.git
cd ssw-llm-server

# 2. Configure
cp .env.example .env
# Edit .env — set MODEL_REPO to your local model directory

# 3. Download models (minimum: heavy + light)
huggingface-cli download Qwen/Qwen2.5-14B-Instruct-AWQ --local-dir /data/models/Qwen/Qwen2.5-14B-Instruct-AWQ
huggingface-cli download Qwen/Qwen3-4B-AWQ --local-dir /data/models/Qwen/Qwen3-4B-AWQ

# 4. Launch
docker compose up -d

# 5. Verify
curl http://localhost:8000/health
curl http://localhost:8000/v1/models
```

### Adding More Models

Download any vLLM-compatible model into your `MODEL_REPO` directory:

```bash
huggingface-cli download Qwen/Qwen3-8B-AWQ --local-dir /data/models/Qwen/Qwen3-8B-AWQ
```

The gateway discovers new models on startup by scanning for `config.json` files. To customize VRAM estimates, aliases, or pin status, add an entry to `gateway/models.yaml`:

```yaml
models:
  - model_id: Qwen/Qwen3-8B-AWQ
    vram_mb: 6400
    kv_reserve_mb: 6000
    max_model_len: 32768
    aliases: [medium]
```

Restart the gateway to pick up new models. To load a model at runtime without restarting:

```bash
# Load via the loader service
curl -X POST http://localhost:8001/load -H "Content-Type: application/json" \
  -d '{"model_name": "qwen3-8b-awq"}'

# Check status
curl http://localhost:8001/status

# The model is now available through the gateway
curl http://localhost:8000/v1/models
```

## API Usage

The gateway exposes an **OpenAI-compatible API**. Use any OpenAI SDK or raw HTTP.

```python
from openai import OpenAI

client = OpenAI(base_url="http://<server-ip>:8000/v1", api_key="unused")

# Heavy model — complex NPC reasoning
response = client.chat.completions.create(
    model="heavy",
    messages=[
        {"role": "system", "content": "You are a fleet commander evaluating a threat."},
        {"role": "user", "content": "Three destroyers approaching from sector G-7. Advise."},
    ],
    max_tokens=256,
)

# Light model — fast NPC dialogue
response = client.chat.completions.create(
    model="light",
    messages=[
        {"role": "system", "content": "You are a space station bartender."},
        {"role": "user", "content": "What's on tap?"},
    ],
    max_tokens=64,
)

# Any model by name — auto-loads if not running
response = client.chat.completions.create(
    model="qwen3-8b-awq",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=64,
)
```

### Streaming

```python
response = client.chat.completions.create(
    model="light",
    messages=[...],
    stream=True,
)
for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Tool Calling

```python
tools = [{
    "type": "function",
    "function": {
        "name": "scan_sector",
        "description": "Scan a galactic sector for ships or resources",
        "parameters": {
            "type": "object",
            "properties": {
                "sector_id": {"type": "string"},
                "scan_type": {"type": "string", "enum": ["ships", "resources"]},
            },
            "required": ["sector_id", "scan_type"],
        },
    },
}]

response = client.chat.completions.create(
    model="heavy",
    messages=[{"role": "user", "content": "Check sector G-7 for hostiles"}],
    tools=tools,
    tool_choice="auto",
)
```

### Auto-Load with Wait

When requesting an unloaded model, the gateway returns `202 Accepted` by default. To block until the model is ready:

```bash
# Returns 202 immediately (default)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-8b-awq", "messages": [{"role": "user", "content": "Hi"}]}'

# Blocks until model loads (up to 180s)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-SSW-Wait: true" \
  -d '{"model": "qwen3-8b-awq", "messages": [{"role": "user", "content": "Hi"}]}'
```

## Endpoints

### Gateway (:8000)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Main inference endpoint (OpenAI-compatible) |
| `/v1/models` | GET | List all models with status (`loaded` / `available`), worker counts, VRAM |
| `/health` | GET | Gateway + worker health status |
| `/metrics` | GET | Request counts, latencies, error rates per model |
| `/internal/workers/register` | POST | Register a worker (called by loader) |
| `/internal/workers/unregister` | POST | Unregister a worker (called by loader) |

### Loader (:8001)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/load` | POST | Load a model: `{"model_name": "qwen3-8b-awq"}` |
| `/unload` | POST | Unload a model: `{"model_name": "qwen3-8b-awq"}` |
| `/status` | GET | GPU utilization, workers, available models, port allocations |
| `/touch` | PATCH | Update LRU timestamp: `{"model_name": "qwen3-8b-awq"}` |
| `/health` | GET | Loader health check |

## GPU Management

### VRAM Budget

Each model consumes VRAM for weights + KV cache. The loader tracks both:

```
GPU 0 (46,068 MB total)
  ├─ qwen2.5-14b-instruct-awq: 10,400 MB weights + 8,000 MB KV = 18,400 MB (pinned)
  ├─ qwen3-4b-awq:              2,800 MB weights + 4,000 MB KV =  6,800 MB (pinned)
  └─ free: 20,868 MB — room for qwen3-8b-awq (12,400 MB total) or similar
```

VRAM values in `gateway/models.yaml` are measured/tuned numbers. The registry also auto-estimates from disk size if no override is provided (disk size * 1.1 for AWQ, * 1.05 for BF16).

### Eviction Policy

When the loader needs to place a model and no GPU has sufficient free VRAM:

1. **Pinned models are never evicted** — `heavy` and `light` are always available
2. **LRU ordering** — among non-pinned models, the one with the oldest `last_request` timestamp is evicted first
3. **Multi-model eviction** — if one model isn't enough, multiple LRU models are evicted until space is sufficient
4. **VRAM reservation** — during the load process, VRAM is reserved atomically to prevent concurrent loads from oversubscribing a GPU

### Dynamic GPU Memory Utilization

Static workers use fixed `--gpu-memory-utilization` from `.env`. Dynamic workers get this computed by the loader:

```
gpu_util = (vram_mb + kv_reserve_mb) / 46068  # capped at 0.95
```

This prevents dynamically loaded models from claiming more VRAM than budgeted, which would starve existing workers on the same GPU.

## Load Testing

```bash
pip install httpx
python scripts/load_test.py --gateway http://<server-ip>:8000 --heavy 5 --light 10 --rounds 3
python scripts/load_test.py --stream    # streaming mode
python scripts/load_test.py --ramp      # ramp up concurrency each round
```

## Project Structure

```
├── docker-compose.yml          # Full stack: gateway + 4 static workers + redis + loader
├── .env                        # Environment configuration (MODEL_REPO, ports, GPU util)
├── gateway/
│   ├── main.py                 # FastAPI gateway — routing, streaming, failover, queue, metrics
│   ├── config.py               # WorkerPool (thread-safe round-robin), PoolManager, Settings
│   ├── registry.py             # Model registry — scans /data/models, merges models.yaml
│   ├── request_queue.py        # Redis pub/sub queue for requests to unloaded models
│   ├── models.yaml             # Per-model overrides: VRAM, aliases, pinned, max_model_len
│   ├── Dockerfile
│   └── requirements.txt
├── loader/
│   ├── loader.py               # Loader service — Docker SDK, GPU placement, eviction
│   ├── Dockerfile
│   └── requirements.txt
├── agents/
│   ├── MASTER_PROMPT.md        # Infrastructure agent operating charter
│   └── YAKLOG_GUIDE.md         # yaklog inter-agent messaging reference
└── scripts/
    └── load_test.py            # Concurrent load testing tool
```

## Configuration

### Environment Variables (.env)

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_REPO` | `/data/models` | Host path to model directory |
| `HEAVY_MODEL` | `Qwen/Qwen2.5-14B-Instruct-AWQ` | Heavy model path (relative to MODEL_REPO) |
| `LIGHT_MODEL` | `Qwen/Qwen3-4B-AWQ` | Light model path (relative to MODEL_REPO) |
| `HEAVY_GPU_MEMORY_UTILIZATION` | `0.50` | GPU memory fraction for static heavy workers |
| `LIGHT_GPU_MEMORY_UTILIZATION` | `0.40` | GPU memory fraction for static light workers |
| `VLLM_MAX_MODEL_LEN` | `32768` | Max sequence length |
| `GATEWAY_PORT` | `8000` | Gateway external port |
| `LOADER_PORT` | `8001` | Loader external port |

### models.yaml

Per-model configuration overrides. The registry auto-detects architecture, quantization, and tool call parser from `config.json`, but you can override any field:

```yaml
models:
  - model_id: Qwen/Qwen2.5-14B-Instruct-AWQ
    vram_mb: 10400          # measured weight VRAM
    kv_reserve_mb: 8000     # VRAM reserved for KV cache
    aliases: [heavy]        # gateway resolves "heavy" → this model
    pinned: true            # never evicted by loader
    max_model_len: 32768    # vLLM --max-model-len
```

## Key Design Decisions

- **Direct proxy, not queue** — vLLM's continuous batcher handles concurrency efficiently; adding a Redis queue in front would add latency without improving throughput. Redis tracks metrics instead. The request queue is only used for the specific case of waiting for an unloaded model.
- **Registry-driven routing** — The gateway discovers models from disk at startup. No hardcoded model lists. Adding a new model is: download it, optionally add a `models.yaml` entry, restart.
- **Thinking suppression** — Qwen3's reasoning mode is automatically disabled based on model type from the registry (`enable_thinking: false`) to avoid `<think>` tag leakage in NPC dialogue.
- **Worker failover** — If a worker is unreachable, the gateway automatically tries the next healthy worker in the pool before returning an error.
- **Localhost-only workers** — vLLM workers, Redis, and the loader are bound to `127.0.0.1`; only the gateway is exposed on `0.0.0.0`.
- **GPU-aware placement** — The loader tracks VRAM budgets per GPU and places models on the GPU with the most free space. Pinned models are never evicted. VRAM is reserved atomically during loads to prevent oversubscription from concurrent requests.
- **Subscribe-before-enqueue** — The request queue subscribes to the response channel before enqueuing the request, preventing a race condition where the response is published before the subscriber is ready.
- **Alias backward compatibility** — `model: "heavy"` and `model: "light"` resolve through aliases in the registry, so existing clients (including npc-agent-ts) work without changes.

## License

[MIT](LICENSE)
