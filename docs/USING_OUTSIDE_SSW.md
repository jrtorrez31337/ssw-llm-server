# SSW AI API — Client Integration Guide

OpenAI-compatible inference API running Qwen3-30B-A3B MoE (128 experts, 8 active per token). Use any OpenAI SDK or HTTP client.

## Endpoint

```
http://192.168.122.76:8000
```

No API key required (internal network only).

## Available Models

| Alias | Routes To | Use Case |
|-------|-----------|----------|
| `light` | Qwen3-30B-A3B MoE AWQ | General purpose, fast |
| `heavy` | Qwen3-30B-A3B MoE AWQ | Same model, same pool |

Both aliases route to the same model. Use either.

To list all available models:

```bash
curl http://192.168.122.76:8000/v1/models
```

## Quick Start

### cURL

```bash
curl http://192.168.122.76:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "light",
    "messages": [{"role": "user", "content": "What is a quasar?"}],
    "max_tokens": 200
  }'
```

### Python (OpenAI SDK)

```bash
pip install openai
```

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://192.168.122.76:8000/v1",
    api_key="unused",
)

response = client.chat.completions.create(
    model="light",
    messages=[{"role": "user", "content": "What is a quasar?"}],
    max_tokens=200,
)

print(response.choices[0].message.content)
```

### TypeScript / Node.js

```bash
npm install openai
```

```typescript
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://192.168.122.76:8000/v1",
  apiKey: "unused",
});

const response = await client.chat.completions.create({
  model: "light",
  messages: [{ role: "user", content: "What is a quasar?" }],
  max_tokens: 200,
});

console.log(response.choices[0].message.content);
```

## Streaming

```python
stream = client.chat.completions.create(
    model="light",
    messages=[{"role": "user", "content": "Explain black holes."}],
    max_tokens: 300,
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## Tool Calling

The model supports OpenAI-format tool calls natively.

```python
response = client.chat.completions.create(
    model="light",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"],
            },
        },
    }],
)

tool_call = response.choices[0].message.tool_calls[0]
print(tool_call.function.name, tool_call.function.arguments)
```

## System Prompts

```python
response = client.chat.completions.create(
    model="light",
    messages=[
        {"role": "system", "content": "You are a concise technical writer."},
        {"role": "user", "content": "Explain DNS in two sentences."},
    ],
    max_tokens=100,
)
```

## Parameters

Standard OpenAI parameters are supported:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `model` | — | Required. Use `light`, `heavy`, or any model name from `/v1/models` |
| `messages` | — | Required. Array of `{role, content}` objects |
| `max_tokens` | model default | Max output tokens |
| `temperature` | 0.7 | 0.0–2.0 |
| `top_p` | 1.0 | Nucleus sampling |
| `stream` | false | SSE streaming |
| `tools` | — | OpenAI-format function definitions |
| `stop` | — | Stop sequences |
| `n` | 1 | Number of completions |

## Context Window

- **Max context**: 65,536 tokens (input + output combined)
- **Typical agent prompts**: ~10,000 tokens input, ~130 tokens output
- Requests exceeding 65K tokens will be truncated

## Error Handling

| Status | Meaning | Action |
|--------|---------|--------|
| 200 | Success | — |
| 202 | Model not loaded | Retry after `Retry-After` header (seconds). Or send `X-SSW-Wait: true` header to block up to 180s |
| 429 | All workers busy | Back off and retry after `Retry-After` header |
| 500 | Server error | Retry with exponential backoff |

## Health Check

```bash
curl http://192.168.122.76:8000/health
```

## Endpoints Reference

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | Chat completions (streaming + non-streaming) |
| GET | `/v1/models` | List available models |
| GET | `/v1/models/{model_id}` | Get specific model info |
| GET | `/v1/models/status` | Model loading status |
| GET | `/health` | Gateway health check |
