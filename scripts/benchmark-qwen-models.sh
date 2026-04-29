#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="$ROOT_DIR/infra/compose/docker-compose.local.yml"
LIMIT="${LIMIT:-8}"
HOST_OUTPUT_DIR="${HOST_OUTPUT_DIR:-$ROOT_DIR/output}"
OUTPUT_PATH="${OUTPUT_PATH:-/app/output/qwen-benchmark-report.json}"

mkdir -p "$HOST_OUTPUT_DIR"

docker compose -f "$COMPOSE_FILE" build inference-worker

docker compose -f "$COMPOSE_FILE" run --rm \
  -e VISION_QWEN_ENABLE_OBJECT_REVIEW=1 \
  -v "$HOST_OUTPUT_DIR:/app/output" \
  inference-worker \
  python /app/scripts/benchmark_qwen_models.py \
  --limit "$LIMIT" \
  --output "$OUTPUT_PATH"
