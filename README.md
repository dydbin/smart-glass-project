# smart-glass-project

Smart-glass monorepo for image capture, VLM inference, memory storage, and app-facing search/chat APIs.

## Repository Layout

```text
smart-glass-project/
|-- apps/
|   |-- api-server/
|   |-- inference-server/
|   |-- admin-web/
|   `-- smart-glass-client/
|-- packages/
|   |-- shared-types/
|   |-- shared-utils/
|   |-- shared-config/
|   `-- ui-kit/
|-- infra/
|   |-- docker/
|   |-- compose/
|   |-- nginx/
|   |-- k8s/
|   `-- terraform/
|-- scripts/
|-- docs/
`-- .github/
```

## Current Architecture

- `api-server` handles capture intake, memory persistence, search, and chat endpoints.
- `inference-server` handles VLM inference and worker execution.
- PostgreSQL stores normalized memory metadata and documents.
- Object storage keeps original captured images.

## Quick Start

```bash
cp .env.example .env
docker compose -f infra/compose/docker-compose.local.yml up --build
```

## Notes

- Put object storage credentials only in `.env`.
- The local compose stack is centered around `api-server`, `inference-server`, `redis`, and `postgres`.
- Search/chat now run inside `api-server`; there is no separate search service in the default flow.
