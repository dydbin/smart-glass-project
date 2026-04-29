#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="$ROOT_DIR/infra/compose/docker-compose.local.yml"
MODEL_KEY="${MODEL_KEY:-qwen2.5-vl-7b}"
IMAGE_PATH="${1:-}"
SKIP_BUILD="${SKIP_BUILD:-1}"

if [[ -z "$IMAGE_PATH" ]]; then
  echo "usage: bash scripts/run-qwen-via-api.sh /absolute/or/relative/path/to/image.jpg" >&2
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
IMAGE_KEY="smoke-tests/qwen-api/local-${SAFE_IMAGE_BASENAME}-${MODEL_KEY}.png"
REQUEST_ID="api-qwen-${SAFE_IMAGE_BASENAME}-${MODEL_KEY}"
API_CONTAINER_IMAGE_PATH="/tmp/local-images/$IMAGE_NAME"

if [[ "$SKIP_BUILD" != "1" ]]; then
  echo "[1/6] Build inference image"
  docker compose -f "$COMPOSE_FILE" build inference-worker inference-api
else
  echo "[1/6] Skip build"
fi

echo "[2/6] Start redis + inference-api + inference-worker"
VISION_CAPTION_MODEL="$MODEL_KEY" \
VISION_CAPTION_QUANTIZATION=4bit \
VISION_CAPTION_DTYPE=float16 \
CELERY_RESULT_BACKEND=redis://redis:6379/1 \
docker compose -f "$COMPOSE_FILE" up -d redis inference-api inference-worker

echo "[3/6] Wait for inference-api readiness"
READY=0
for attempt in {1..30}; do
  if docker compose -f "$COMPOSE_FILE" exec -T \
    inference-api \
    python /app/scripts/http_healthcheck.py http://127.0.0.1:8000/health/ready; then
    READY=1
    break
  fi
  echo "api readiness not ready yet, retrying (${attempt}/30)"
  sleep 3
done

if [[ "$READY" -ne 1 ]]; then
  echo "inference-api did not become ready in time. Recent logs:" >&2
  docker compose -f "$COMPOSE_FILE" logs --tail=200 inference-api >&2 || true
  exit 1
fi

echo "[4/6] Copy local image into running inference-api and upload to object storage"
docker compose -f "$COMPOSE_FILE" exec -T inference-api mkdir -p /tmp/local-images
docker compose -f "$COMPOSE_FILE" cp "$ABS_IMAGE_PATH" "inference-api:$API_CONTAINER_IMAGE_PATH"
docker compose -f "$COMPOSE_FILE" exec -T \
  inference-api \
  python /app/scripts/upload_local_image_to_s3.py \
  --image "$API_CONTAINER_IMAGE_PATH" \
  --image-key "$IMAGE_KEY"

echo "[5/6] Enqueue inference task via API"
ENQUEUE_RESPONSE="$(
python - <<PY
import json
import urllib.request

payload = {
    "imageKey": "$IMAGE_KEY",
    "userId": "smoke-user",
    "memoryId": "smoke-memory",
    "requestId": "$REQUEST_ID",
    "taskType": "metadata",
}
req = urllib.request.Request(
    "http://127.0.0.1:8000/tasks/vision",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=30) as response:
    body = json.loads(response.read().decode("utf-8"))
print(json.dumps(body))
PY
)"
TASK_ID="$(
python - <<PY
import json
body = json.loads('''$ENQUEUE_RESPONSE''')
print(body["taskId"])
PY
)"
STATUS_URL="$(
python - <<PY
import json
body = json.loads('''$ENQUEUE_RESPONSE''')
print(body.get("statusUrl") or f"/tasks/{body['taskId']}")
PY
)"
echo "task_id=$TASK_ID"
echo "status_url=$STATUS_URL"

echo "[6/6] Poll task result via API"
python - <<PY
import json
import sys
import time
import urllib.request

task_id = "$TASK_ID"
status_url = "$STATUS_URL"
url = f"http://127.0.0.1:8000{status_url}"

for attempt in range(1, 61):
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    status = payload.get("taskStatus")
    state = payload.get("state")
    print(f"poll {attempt}/60: taskStatus={status} state={state}", file=sys.stderr)

    if status == "completed":
        result = payload.get("result") or {}
        if result.get("status") != "success":
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            raise SystemExit("Task completed without success result")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(0)

    if status == "failed":
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        raise SystemExit("Task failed")

    time.sleep(5)

raise SystemExit("Timed out waiting for task completion")
PY
