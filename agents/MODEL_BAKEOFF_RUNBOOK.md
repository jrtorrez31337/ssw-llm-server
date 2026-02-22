# Model Bakeoff Runbook

Standalone bakeoff stack (separate compose project, ports, network, redis volume). It is safe to run alongside the current stack.

## Files

- `docker-compose.bakeoff.yml`
- `.env.bakeoff`
- `gateway/models.bakeoff.yaml`

## Validate (no changes to running containers)

```bash
docker compose -f docker-compose.bakeoff.yml --env-file .env.bakeoff config >/tmp/bakeoff.compose.rendered.yml
```

## Start bakeoff stack

```bash
docker compose -f docker-compose.bakeoff.yml --env-file .env.bakeoff up -d
```

Endpoints:

- Gateway: `http://127.0.0.1:8100`
- Loader: `http://127.0.0.1:8111`

## Load baseline + candidate

`models.bakeoff.yaml` defaults:

- `light` alias -> `Qwen/Qwen3-4B-AWQ` (baseline)
- `heavy` + `candidate` aliases -> `Qwen/Qwen3-4B-Instruct-2507-AWQ` (first challenger)

Load both pools:

```bash
curl -sS -X POST http://127.0.0.1:8111/load -H 'Content-Type: application/json' \
  -d '{"model_name":"light"}'

curl -sS -X POST http://127.0.0.1:8111/load -H 'Content-Type: application/json' \
  -d '{"model_name":"heavy"}'
```

Verify:

```bash
curl -sS http://127.0.0.1:8100/health
curl -sS http://127.0.0.1:8100/v1/models/status
```

## Switch challenger model

1. Move `aliases: [heavy, candidate]` in `gateway/models.bakeoff.yaml` to the next model.
2. Reload gateway + loader config:

```bash
docker compose -f docker-compose.bakeoff.yml --env-file .env.bakeoff up -d --build gateway loader
```

3. Unload old challenger if loaded, then load new challenger:

```bash
curl -sS -X POST http://127.0.0.1:8111/unload -H 'Content-Type: application/json' \
  -d '{"model_name":"heavy"}'

curl -sS -X POST http://127.0.0.1:8111/load -H 'Content-Type: application/json' \
  -d '{"model_name":"heavy"}'
```

## Recommended challenger order

1. `Qwen/Qwen3-4B-Instruct-2507-AWQ`
2. `Qwen/Qwen3-8B-AWQ`
3. `meta-llama/Llama-3.1-8B-Instruct`
4. `mistralai/Mistral-7B-Instruct-v0.2`

## Promotion gates

- Latency: p95 <= 20s, max <= 40s
- Stability: no sustained 503/no-healthy-worker periods
- Behavior: no regression in stuck overrides or scan->action conversion
