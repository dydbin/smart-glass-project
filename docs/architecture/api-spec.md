# API Spec

## Purpose

This document describes the current API contract between:

- `api-server`
- `inference-server`
- PostgreSQL-backed memory storage inside `api-server`

There is no separate search service in the default application flow anymore.

## Capture Flow

1. A client uploads the original image to object storage.
2. The client registers the capture with `POST /media/captures`.
3. `api-server` dispatches the inference worker.
4. The worker returns a normalized VLM result.
5. `api-server` stores the normalized memory document in PostgreSQL.

## Main Endpoints

### `POST /media/captures`

Registers a captured image, runs the inference worker, and persists the result when memory storage is enabled.

Relevant request fields:

- `captureId`
- `requestId`
- `memoryId`
- `userId`
- `taskType`
- `capturedAt`
- `fileName`
- `imageKey`
- `imageUrl`
- `contentType`
- `sourceImage`
- `generation`

Relevant response sections:

- `capture`
- `worker`
- `memoryStore`

### `POST /search`

Searches stored memories for one user.

Request:

```json
{
  "userId": "user-1",
  "query": "my wallet",
  "topK": 5
}
```

The endpoint also accepts the legacy snake_case aliases:

```json
{
  "user_id": "user-1",
  "query": "my wallet",
  "top_k": 5
}
```

Response fields:

- `query`
- `totalHits`
- `hits[]`

Each hit includes:

- `memoryId`
- `score`
- `lexicalScore`
- `matchedTerms`
- `imageKey`
- `imageUrl`
- `capturedAt`
- `caption`
- `sceneSummary`
- `positionHint`
- `location`
- `detectedObjects`
- `tags`

### `POST /chat`

Builds a grounded answer from the current search results.

Request:

```json
{
  "userId": "user-1",
  "query": "where was my wallet?",
  "topK": 3
}
```

Response fields:

- `answer`
- `answerMode`
- `query`
- `totalHits`
- `hits[]`
- `citedMemoryIds`
- `confidence`
- `reason`

The current default answer mode is template-based and does not require a separate LLM service.

## VLM Result Contract

The worker result is normalized around:

- `status`
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

Error results keep `errorCode`, `message`, and `retryable` for backward
compatibility, and also include `errorDetails` for diagnostics. The detail
payload carries a stable failure category, normalized reason, exception type,
retryability, source, and task time-limit values when available.

Current reference files:

- [apps/inference-server/src/contracts/vlm.py](/home/ghpark/projects/smart-glass-project/apps/inference-server/src/contracts/vlm.py)
- [apps/inference-server/src/queue/tasks.py](/home/ghpark/projects/smart-glass-project/apps/inference-server/src/queue/tasks.py)
- [apps/api-server/src/api/schemas.py](/home/ghpark/projects/smart-glass-project/apps/api-server/src/api/schemas.py)
- [apps/api-server/src/database/memory_store.py](/home/ghpark/projects/smart-glass-project/apps/api-server/src/database/memory_store.py)

## Storage Notes

- Original images stay in object storage.
- Normalized memory documents are stored in PostgreSQL as JSONB.
- Search is currently based on recent-history retrieval plus keyword/location matching.
- A separate vector database is not part of the default architecture.
