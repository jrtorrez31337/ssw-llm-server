# SSW NPC Agent — AI Gateway Integration Guide

You are building NPC agents for Social Space Wars that use the SSW AI inference platform for decision-making, dialogue, and behavior generation. This document is your interface contract.

---

## Endpoint

```
POST http://<GATEWAY_HOST>:8000/v1/chat/completions
```

The API is **OpenAI-compatible**. Use any OpenAI SDK or raw HTTP.

---

## Models

| Model ID | Backend | Use When |
|----------|---------|----------|
| `heavy`  | Qwen2.5-14B-Instruct-AWQ | Complex reasoning, multi-step planning, diplomatic negotiations, lore generation, quest design |
| `light`  | Qwen3-8B-AWQ | Ambient dialogue, simple reactions, status checks, routine NPC behaviors, high-frequency calls |

**Rule of thumb**: if the player won't notice the quality difference, use `light`. It's faster and leaves `heavy` capacity for tasks that matter.

---

## Basic Request

```json
{
  "model": "light",
  "messages": [
    {"role": "system", "content": "You are a merchant NPC in a space station..."},
    {"role": "user", "content": "What do you have for sale?"}
  ],
  "max_tokens": 256,
  "temperature": 0.7
}
```

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(base_url="http://<GATEWAY_HOST>:8000/v1", api_key="unused")

response = client.chat.completions.create(
    model="light",
    messages=[
        {"role": "system", "content": "You are a merchant NPC..."},
        {"role": "user", "content": "What do you have for sale?"},
    ],
    max_tokens=256,
    temperature=0.7,
)
print(response.choices[0].message.content)
```

### Python (httpx / raw HTTP)

```python
import httpx

resp = httpx.post(
    "http://<GATEWAY_HOST>:8000/v1/chat/completions",
    json={
        "model": "heavy",
        "messages": [{"role": "user", "content": "Analyze threat level of incoming fleet"}],
        "max_tokens": 512,
    },
)
print(resp.json()["choices"][0]["message"]["content"])
```

---

## Streaming

For real-time dialogue or long outputs, use streaming to get tokens as they generate:

```python
response = client.chat.completions.create(
    model="light",
    messages=[...],
    max_tokens=256,
    stream=True,
)
for chunk in response:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
```

---

## Tool Calling

Both models support tool/function calling via the Hermes format. Define tools and let the model decide when to invoke them.

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "scan_sector",
            "description": "Scan a galactic sector for ships, resources, or anomalies",
            "parameters": {
                "type": "object",
                "properties": {
                    "sector_id": {"type": "string", "description": "Sector coordinate ID"},
                    "scan_type": {"type": "string", "enum": ["ships", "resources", "anomalies"]},
                },
                "required": ["sector_id", "scan_type"],
            },
        },
    }
]

response = client.chat.completions.create(
    model="heavy",
    messages=[{"role": "user", "content": "Check sector G-7 for hostile ships"}],
    tools=tools,
    tool_choice="auto",
)

msg = response.choices[0].message
if msg.tool_calls:
    for call in msg.tool_calls:
        print(f"Call: {call.function.name}({call.function.arguments})")
```

After executing the tool, send the result back:

```python
messages.append(msg)  # assistant message with tool_calls
messages.append({
    "role": "tool",
    "tool_call_id": call.id,
    "content": '{"ships": [{"id": "X-42", "class": "destroyer", "hostile": true}]}',
})
# Then call completions again for the model to interpret the result
```

---

## Model Selection Guidelines

### Use `heavy` for:
- Multi-turn strategic reasoning (fleet commander AI, faction leader diplomacy)
- Content generation (quest narratives, item descriptions, lore entries)
- Complex tool-calling chains (multi-step plans with 3+ tool invocations)
- Situations where coherence across a long context window matters

### Use `light` for:
- Single-turn NPC barks and ambient dialogue
- Simple decisions (fight/flee, buy/sell, patrol/idle)
- High-frequency background agent ticks (economy bots, patrol AI)
- Any call where latency matters more than depth

### Anti-patterns
- Do NOT send `heavy` requests for one-line NPC dialogue — wastes capacity
- Do NOT set `max_tokens` higher than you need — longer generation = longer latency
- Do NOT poll in tight loops — batch or debounce your requests
- Do NOT include the entire game state in every message — send only what the NPC needs to know

---

## Request Parameters Reference

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `model` | string | required | `"heavy"` or `"light"` |
| `messages` | array | required | OpenAI message format |
| `max_tokens` | int | model default | Keep as low as practical |
| `temperature` | float | 1.0 | 0.3-0.5 for deterministic, 0.7-1.0 for creative |
| `top_p` | float | 1.0 | Alternative to temperature — don't use both |
| `stream` | bool | false | Enable SSE streaming |
| `tools` | array | none | Tool/function definitions |
| `tool_choice` | string/object | "auto" | `"auto"`, `"none"`, or specific function |
| `stop` | string/array | none | Stop sequences |

---

## Health & Diagnostics

```bash
# Gateway health (are workers up?)
curl http://<GATEWAY_HOST>:8000/health

# Available models
curl http://<GATEWAY_HOST>:8000/v1/models

# Request metrics (counts, latencies, error rates)
curl http://<GATEWAY_HOST>:8000/metrics
```

Check `/health` before entering a request loop. If status is `degraded`, some workers are down — requests will still route to healthy workers but capacity is reduced.

---

## Error Handling

| Status | Meaning | Action |
|--------|---------|--------|
| 200 | Success | Process response |
| 503 | No healthy workers / unknown model | Retry after delay, check `/health` |
| 502 | Worker unreachable after failover | Back off, alert ops |
| 422 | Malformed request | Fix request body |

Implement exponential backoff on 502/503. The gateway already does one failover attempt internally — if you get an error, all workers in that pool are likely down.

---

## Context Window

Both models are configured with an **8192 token context window**. Budget your system prompt + conversation history + max_tokens to stay within this limit. If you exceed it, the request will fail.

Rough token math: ~4 characters per token for English text.

---

## Performance Expectations

- **Light model**: ~50-150ms time-to-first-token, suitable for real-time NPC interaction
- **Heavy model**: ~100-300ms time-to-first-token, use for background/async reasoning
- The gateway round-robins across 2 replicas per model — concurrent requests distribute automatically
- Under burst load, requests queue inside vLLM's continuous batcher — latency increases but requests don't drop
