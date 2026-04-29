# System Architecture

- `apps/api-server`: main application API, capture intake, memory persistence, search, and chat
- `apps/inference-server`: VLM inference API and Celery worker
- `apps/admin-web`: admin interface
- `apps/smart-glass-client`: smart-glass client skeleton
- `packages/*`: shared types, utils, config, and UI assets
- `infra/*`: Docker, Compose, Nginx, Kubernetes, and Terraform assets
