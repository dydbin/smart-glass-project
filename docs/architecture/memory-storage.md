# Memory Storage

This project uses PostgreSQL as the durable memory store behind `api-server`.

## What Is Stored

- `memory_id`
- `user_id`
- `image_key` / `image_url`
- `captured_at`
- `caption`, `scene_summary`, `ocr_text`, `position_hint`
- `detected_objects`, `tags`
- `location`
- the normalized memory document as JSONB

## Local Docker Workflow

```powershell
Copy-Item .env.example .env
docker compose -f infra/compose/docker-compose.local.yml up --build
```

The default local compose stack starts:

- `postgres`
- `redis`
- `inference-api`
- `inference-worker`
- `api-server`

Successful VLM results are persisted by `api-server` into PostgreSQL automatically in this setup.

## Notes

- Search and chat are served directly by `api-server`.
- Original images stay in object storage.
- The current query flow is recent-history retrieval plus keyword/location matching, without a separate vector database.
