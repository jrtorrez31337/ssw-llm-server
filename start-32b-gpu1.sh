#!/usr/bin/env bash
# Start SSW AI inference (Qwen3-32B-AWQ on GPU 1 only) + observability.
# GPU 0 remains free for other workloads.
#
#   Usage:  ./start-32b-gpu1.sh [--no-obs]

set -euo pipefail
cd "$(dirname "$0")"

OBS=true
for arg in "$@"; do
  [[ "$arg" == "--no-obs" ]] && OBS=false
done

echo "==> Starting vLLM stack  (Qwen3-32B-AWQ × 1 worker, GPU 1 only)..."
docker compose \
  -p sswai \
  -f docker-compose.32b-gpu1.yml \
  --env-file .env.32b-gpu1 \
  up -d

if $OBS; then
  echo "==> Starting observability stack  (Prometheus + Grafana)..."
  docker compose \
    -p sswai \
    -f docker-compose.observability.yml \
    up -d
fi

echo ""
echo "Stack ready:"
echo "  Gateway     http://localhost:8000  (1 worker, GPU 1)"
echo "  GPU 0       FREE"
if $OBS; then
  echo "  Prometheus  http://localhost:9090"
  echo "  Grafana     http://localhost:3000  (admin / sswai)"
fi
