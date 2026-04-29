#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="$ROOT_DIR/infra/compose/docker-compose.local.yml"
MODEL_KEY="${MODEL_KEY:-qwen2.5-vl-3b}"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker command not found. Enable Docker Desktop WSL integration first." >&2
  exit 1
fi

if [[ ! -f "$ROOT_DIR/.env" ]]; then
  echo ".env file is required. Copy from .env.example and fill object storage credentials." >&2
  exit 1
fi

echo "[1/4] Build inference image"
docker compose -f "$COMPOSE_FILE" build inference-worker inference-api

echo "[2/4] Start redis + inference-api"
VISION_CAPTION_MODEL="$MODEL_KEY" \
VISION_CAPTION_QUANTIZATION=4bit \
VISION_CAPTION_DTYPE=float16 \
docker compose -f "$COMPOSE_FILE" up -d redis inference-api

echo "[3/4] Readiness probe"
READY=0
for attempt in {1..20}; do
  if docker compose -f "$COMPOSE_FILE" exec -T \
    -e VISION_CAPTION_MODEL="$MODEL_KEY" \
    -e VISION_CAPTION_QUANTIZATION=4bit \
    -e VISION_CAPTION_DTYPE=float16 \
    inference-api \
    python /app/scripts/http_healthcheck.py http://127.0.0.1:8000/health/ready; then
    READY=1
    break
  fi
  echo "readiness not ready yet, retrying (${attempt}/20)"
  sleep 3
done

if [[ "$READY" -ne 1 ]]; then
  echo "inference-api did not become ready in time. Recent logs:" >&2
  docker compose -f "$COMPOSE_FILE" logs --tail=200 inference-api >&2 || true
  exit 1
fi

echo "[4/4] Worker smoke test with sample image"
docker compose -f "$COMPOSE_FILE" run --rm \
  -e VISION_CAPTION_MODEL="$MODEL_KEY" \
  -e VISION_CAPTION_QUANTIZATION=4bit \
  -e VISION_CAPTION_DTYPE=float16 \
  inference-worker \
  python /app/scripts/smoke_qwen_e2e.py --image /app/sample_data/key_1.jpg --model-key "$MODEL_KEY"

echo "Smoke test completed."
