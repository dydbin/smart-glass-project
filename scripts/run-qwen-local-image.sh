#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="$ROOT_DIR/infra/compose/docker-compose.local.yml"
MODEL_KEY="${MODEL_KEY:-qwen2.5-vl-7b}"
IMAGE_PATH="${1:-}"
SKIP_BUILD="${SKIP_BUILD:-1}"

if [[ -z "$IMAGE_PATH" ]]; then
  echo "usage: bash scripts/run-qwen-local-image.sh /absolute/or/relative/path/to/image.jpg" >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker command not found. Enable Docker Desktop WSL integration first." >&2
  exit 1
fi

if [[ ! -f "$ROOT_DIR/.env" ]]; then
  echo ".env file is required. Copy from .env.example and fill object storage credentials." >&2
  exit 1
fi

ABS_IMAGE_PATH="$(realpath "$IMAGE_PATH")"
if [[ ! -f "$ABS_IMAGE_PATH" ]]; then
  echo "image file not found: $ABS_IMAGE_PATH" >&2
  exit 1
fi

IMAGE_NAME="$(basename "$ABS_IMAGE_PATH")"
IMAGE_BASENAME="${IMAGE_NAME%.*}"
SAFE_IMAGE_BASENAME="$(printf '%s' "$IMAGE_BASENAME" | tr ' /' '__')"
IMAGE_KEY="smoke-tests/qwen/local-${SAFE_IMAGE_BASENAME}-${MODEL_KEY}.jpg"
REQUEST_ID="local-qwen-${SAFE_IMAGE_BASENAME}-${MODEL_KEY}"
CONTAINER_IMAGE_PATH="/tmp/local-images/$IMAGE_NAME"

if [[ "$SKIP_BUILD" != "1" ]]; then
  echo "[1/5] Build inference image"
  docker compose -f "$COMPOSE_FILE" build inference-worker inference-api
else
  echo "[1/5] Skip build"
fi

echo "[2/5] Start redis + inference-api + inference-worker"
VISION_CAPTION_MODEL="$MODEL_KEY" \
VISION_CAPTION_QUANTIZATION=4bit \
VISION_CAPTION_DTYPE=float16 \
docker compose -f "$COMPOSE_FILE" up -d redis inference-api inference-worker

echo "[3/5] Readiness probe"
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

echo "[4/5] Copy local image into running inference-worker"
docker compose -f "$COMPOSE_FILE" exec -T inference-worker mkdir -p /tmp/local-images
docker compose -f "$COMPOSE_FILE" cp "$ABS_IMAGE_PATH" "inference-worker:$CONTAINER_IMAGE_PATH"

echo "[5/5] Worker inference with local image"
docker compose -f "$COMPOSE_FILE" exec -T \
  -e VISION_CAPTION_MODEL="$MODEL_KEY" \
  -e VISION_CAPTION_QUANTIZATION=4bit \
  -e VISION_CAPTION_DTYPE=float16 \
  inference-worker \
  python /app/scripts/smoke_qwen_e2e.py \
  --image "$CONTAINER_IMAGE_PATH" \
  --image-key "$IMAGE_KEY" \
  --request-id "$REQUEST_ID" \
  --model-key "$MODEL_KEY"

echo "Local image inference completed."
