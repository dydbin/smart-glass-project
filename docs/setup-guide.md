# Setup Guide

## Local Development

The recommended local workflow uses Docker Compose from the repository root.

```powershell
Copy-Item .env.example .env
docker compose -f infra/compose/docker-compose.local.yml up --build
```

The default local stack starts:

- `postgres`
- `redis`
- `inference-api`
- `inference-worker`
- `api-server`

## Service Roles

- `api-server`: capture registration, PostgreSQL memory persistence, search, and chat APIs
- `inference-server`: VLM inference API and Celery worker
- `postgres`: durable memory store for normalized metadata
- `redis`: queue broker/result backend

## Required Environment Values

Set these in `.env` before running the stack:

- `STORAGE_ACCESS_KEY_ID`
- `STORAGE_SECRET_ACCESS_KEY`
- `STORAGE_BUCKET_NAME`
- `STORAGE_REGION`
- `STORAGE_ENDPOINT_URL`
- `STORAGE_ADDRESSING_STYLE`

The default object storage settings target Naver Object Storage:

- `STORAGE_REGION=kr-standard`
- `STORAGE_ENDPOINT_URL=https://kr.object.ncloudstorage.com`
- `STORAGE_ADDRESSING_STYLE=path`

## Optional Environment Values

- `API_CAPTURE_ENABLE_MEMORY_STORE`
- `API_CAPTURE_DATABASE_URL`
- `API_MEMORY_DEFAULT_TOP_K`
- `VISION_QWEN_FALLBACK_MODEL`
- `VISION_QWEN_ENABLE_OBJECT_REVIEW`

## WSL / DNS Notes

If name resolution to `kr.object.ncloudstorage.com` fails inside WSL, object storage probes and CLI checks can fail even when Docker containers are healthy.

Useful checks:

```bash
getent hosts kr.object.ncloudstorage.com
python -c "import socket; print(socket.getaddrinfo('kr.object.ncloudstorage.com', 443))"
cat /etc/resolv.conf
```

If needed, disable auto-generated WSL DNS and write `/etc/resolv.conf` manually:

```ini
[network]
generateResolvConf = false
```

```bash
wsl --shutdown
sudo rm -f /etc/resolv.conf
printf "nameserver 1.1.1.1\nnameserver 8.8.8.8\n" | sudo tee /etc/resolv.conf
```

## Postgres Volume Note

If you already ran an older local stack with different Postgres credentials, recreate the local `postgres-data` volume before starting the updated compose stack.
