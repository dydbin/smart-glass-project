# VLM Input/Output Schema

## Goal

The project keeps a model-agnostic VLM result contract so that:

- `inference-server` can evolve independently
- `api-server` can persist normalized memory documents
- downstream search/chat APIs can stay stable even if the underlying VLM changes

## Input Shape

The inference request is built around:

- `requestId`
- `taskType`
- `memoryId`
- `userId`
- `capturedAt`
- `sourceImage`
- `generation`

Reference:

- [apps/api-server/src/api/schemas.py](/home/ghpark/projects/smart-glass-project/apps/api-server/src/api/schemas.py)

## Success Result Shape

Successful worker results contain:

- `status: "success"`
- `requestId`
- `taskType`
- `memoryId`
- `userId`
- `capturedAt`
- `sourceImage`
- `metadata`
- `pipelineOutput`
- `providerMetadata`
- `runtime`

Important `metadata` fields:

- `caption`
- `sceneSummary`
- `detectedObjects`
- `tags`
- `ocrText`
- `positionHint`
- `location`

## Error Result Shape

Error results contain:

- `status: "error"`
- `requestId`
- `taskType`
- `memoryId`
- `userId`
- `capturedAt`
- `sourceImage`
- `errorCode`
- `message`
- `retryable`
- `errorDetails`
- `providerMetadata`

`errorDetails` is intended for diagnostics and operator decisions. It includes:

- `category`: broad failure class such as `timeout`, `storage`, `source_image`, `input`, `model`, or `unknown`
- `reason`: normalized machine-readable reason, usually matching `errorCode`
- `exceptionType`: exception class name without stack trace
- `retryable`: the same retry decision as the top-level field
- `source`: component that produced the detail payload, when known
- `taskTimeLimit`: `softSec` and `hardSec` values for worker timeout context, when available

## Memory Normalization

`api-server` converts a successful VLM result into a normalized memory document before writing to PostgreSQL.

Normalization responsibilities include:

- requiring `memoryId` and `userId`
- normalizing `imageKey`, `imageUrl`, `capturedAt`
- normalizing `caption`, `sceneSummary`, `ocrText`, `positionHint`
- deduplicating `detectedObjects` and `tags`
- deriving location hints from `pipelineOutput` when needed

Reference:

- [apps/api-server/src/database/memory_store.py](/home/ghpark/projects/smart-glass-project/apps/api-server/src/database/memory_store.py)

## Verification

Current lightweight verification paths:

- `PYTHONPATH=apps/api-server python -m py_compile apps/api-server/src/database/memory_store.py`
- `PYTHONPATH=apps/api-server python -m unittest apps/api-server/tests/test_memory_store.py`
- `PYTHONPATH=apps/api-server python -m unittest apps/api-server/tests/test_search_service.py`

## Current Architectural Position

- Original images are stored in object storage.
- VLM execution stays in `inference-server`.
- Memory persistence, search, and chat APIs are handled in `api-server`.
- The default application path no longer depends on a separate search service.
