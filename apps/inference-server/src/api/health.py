from datetime import datetime, timezone
from typing import Any, Dict, Tuple

from src.health.checks import (
    build_health_summary,
    check_model_config as _check_model_config,
    check_queue as _check_queue,
    check_storage as _check_storage,
)


def build_liveness_payload() -> Tuple[int, Dict[str, Any]]:
    checks = {
        "api": {"status": "ok"},
    }
    return 200, {
        "status": "ok",
        "service": "inference-server",
        "check_type": "liveness",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": build_health_summary(checks),
        "checks": checks,
    }


def build_health_payload() -> Tuple[int, Dict[str, Any]]:
    _, queue_detail = _check_queue()
    _, storage_detail = _check_storage()
    _, model_detail = _check_model_config()

    checks = {
        "api": {"status": "ok"},
        "queue": queue_detail,
        "storage": storage_detail,
        "model": model_detail,
    }
    summary = build_health_summary(checks)
    ready = summary["failing"] == 0
    overall_status = "ok" if ready else "degraded"
    status_code = 200 if ready else 503

    payload = {
        "status": overall_status,
        "service": "inference-server",
        "check_type": "readiness",
        "ready": ready,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "checks": checks,
    }
    return status_code, payload
