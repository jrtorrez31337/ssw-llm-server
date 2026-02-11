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
  POST /v1/chat/completions  (model: "heavy" or "light")
  │
  ▼
FastAPI Gateway (:8000)  ─── 0.0.0.0, accepts remote clients
  │
  ├─ model="heavy" → round-robin [heavy-0, heavy-1]  (Qwen2.5-14B-Instruct-AWQ)
  └─ model="light" → round-robin [light-0, light-1]  (Qwen3-8B-AWQ)
       │
       ▼
  vLLM Workers (continuous batching, tool calling)
       │
  Redis ← request metrics (latency, tokens, errors)
```

### Two-Tier Model Strategy

| Model | Backend | Use Case |
|-------|---------|----------|
| `heavy` | Qwen2.5-14B-Instruct-AWQ | Complex reasoning, diplomacy, quest generation, multi-step planning |
| `light` | Qwen3-8B-AWQ | Ambient dialogue, simple reactions, high-frequency NPC ticks |

Both models support **tool/function calling** via the Hermes format, enabling NPCs to interact with game systems (scan sectors, check inventories, send messages) through structured tool invocations.

### Hardware Requirements

- **2x NVIDIA GPUs** with 24GB+ VRAM each (tested on A40 48GB)
- Each GPU runs 1x 14B worker (~9GB) + 1x 8B worker (~6GB)
- Remaining VRAM used for KV cache (concurrent request batching)

## Quick Start

```bash
# 1. Clone
git clone https://github.com/jrtorrez31337/ssw-llm-server.git
cd ssw-llm-server

# 2. Configure
cp .env.example .env
# Edit .env — set MODEL_REPO to your local model directory

# 3. Download models
# Models must be pre-downloaded to your MODEL_REPO path
huggingface-cli download Qwen/Qwen2.5-14B-Instruct-AWQ --local-dir /data/models/Qwen/Qwen2.5-14B-Instruct-AWQ
huggingface-cli download Qwen/Qwen3-8B-AWQ --local-dir /data/models/Qwen/Qwen3-8B-AWQ

# 4. Launch
docker compose up -d

# 5. Verify
curl http://localhost:8000/health
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

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Main inference endpoint (OpenAI-compatible) |
| `/v1/models` | GET | List available models |
| `/health` | GET | Gateway + worker health status |
| `/metrics` | GET | Request counts, latencies, error rates |

## Load Testing

```bash
pip install httpx
python scripts/load_test.py --gateway http://<server-ip>:8000 --heavy 5 --light 10 --rounds 3
python scripts/load_test.py --stream    # streaming mode
python scripts/load_test.py --ramp      # ramp up concurrency each round
```

## Project Structure

```
├── docker-compose.yml      # Full stack orchestration (gateway + 4 workers + redis)
├── .env.example            # Environment configuration template
├── gateway/
│   ├── main.py             # FastAPI gateway — routing, streaming, metrics
│   ├── config.py           # Worker pool config, health tracking
│   ├── Dockerfile
│   └── requirements.txt
├── agents/
│   ├── MASTER_PROMPT.md    # Infrastructure agent operating charter
│   └── NPC_AGENT_GUIDE.md  # NPC client integration reference
└── scripts/
    └── load_test.py        # Concurrent load testing tool
```

## Key Design Decisions

- **Direct proxy, not queue** — vLLM's continuous batcher handles concurrency efficiently; adding a Redis queue in front would add latency without improving throughput. Redis tracks metrics instead.
- **Thinking suppression** — Qwen3's reasoning mode is automatically disabled for light model requests (`enable_thinking: false`) to avoid `<think>` tag leakage in NPC dialogue.
- **Worker failover** — If a worker is unreachable, the gateway automatically tries the next healthy worker in the pool before returning an error.
- **Localhost-only workers** — vLLM workers and Redis are bound to `127.0.0.1`; only the gateway is exposed on `0.0.0.0`.

## License

[MIT](LICENSE)
