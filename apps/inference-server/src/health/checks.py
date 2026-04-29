import os
import socket
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import redis
from celery import Celery

from src.models.registry import resolve_inference_model
from src.models.serving_profile import resolve_runtime_execution_policy
from src.storage.s3 import StorageAccessError, StorageConfigError, get_storage_service
from src.worker_preload import get_preload_status_path


def _extract_check_status(detail: Mapping[str, Any]) -> str:
    raw_status = detail.get("status")
    normalized = str(raw_status or "").strip().lower()
    return normalized or "unknown"


def build_health_summary(checks: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    statuses = {
        name: _extract_check_status(detail)
        for name, detail in checks.items()
    }
    failing_checks = [
        name
        for name, status in statuses.items()
        if status != "ok"
    ]
    return {
        "total": len(statuses),
        "passing": len(statuses) - len(failing_checks),
        "failing": len(failing_checks),
        "failingChecks": failing_checks,
        "statuses": statuses,
    }


def check_queue() -> Tuple[str, Dict[str, Any]]:
    broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    detail: Dict[str, Any] = {"broker_url": broker_url}

    try:
        client = redis.Redis.from_url(
            broker_url, socket_connect_timeout=1, socket_timeout=1
        )
        client.ping()
        detail["status"] = "ok"
    except Exception as exc:
        detail["status"] = "error"
        detail["message"] = str(exc)

    return detail["status"], detail


def check_storage_config() -> Tuple[str, Dict[str, Any]]:
    region = (
        os.getenv("STORAGE_REGION")
        or os.getenv("AWS_REGION")
        or "ap-northeast-2"
    ).strip() or "ap-northeast-2"
    bucket_name = (
        os.getenv("STORAGE_BUCKET_NAME") or os.getenv("AWS_S3_BUCKET_NAME") or ""
    ).strip()
    missing = [
        name
        for name, aliases in (
            ("STORAGE_ACCESS_KEY_ID", ("STORAGE_ACCESS_KEY_ID", "AWS_ACCESS_KEY_ID")),
            (
                "STORAGE_SECRET_ACCESS_KEY",
                ("STORAGE_SECRET_ACCESS_KEY", "AWS_SECRET_ACCESS_KEY"),
            ),
            ("STORAGE_BUCKET_NAME", ("STORAGE_BUCKET_NAME", "AWS_S3_BUCKET_NAME")),
        )
        if not any(os.getenv(alias, "").strip() for alias in aliases)
    ]

    detail: Dict[str, Any] = {
        "region": region,
        "bucket_name": bucket_name or None,
        "endpoint_url": os.getenv("STORAGE_ENDPOINT_URL", "").strip() or None,
    }
    if missing:
        detail["status"] = "error"
        detail["missing_env"] = missing
    else:
        detail["status"] = "ok"

    return detail["status"], detail


def _resolve_storage_readiness_mode() -> str:
    return (
        os.getenv("INFERENCE_STORAGE_READINESS_MODE", "config").strip().lower()
        or "config"
    )


def check_storage() -> Tuple[str, Dict[str, Any]]:
    config_status, config_detail = check_storage_config()
    mode = _resolve_storage_readiness_mode()
    detail: Dict[str, Any] = {
        **config_detail,
        "mode": mode,
    }

    if mode not in {"config", "deep"}:
        detail["status"] = "error"
        detail["message"] = (
            "Unsupported INFERENCE_STORAGE_READINESS_MODE; use 'config' or 'deep'"
        )
        return detail["status"], detail

    if config_status != "ok":
        detail["probe"] = {
            "status": "skipped",
            "reason": "storage configuration is incomplete",
        }
        return detail["status"], detail

    if mode == "config":
        detail["status"] = "ok"
        detail["probe"] = {
            "status": "skipped",
            "reason": "storage readiness mode is config",
        }
        return detail["status"], detail

    try:
        get_storage_service().probe_bucket_access(
            bucket_name=detail.get("bucket_name"),
        )
    except (StorageConfigError, StorageAccessError) as exc:
        detail["status"] = "error"
        detail["probe"] = {
            "status": "error",
            "message": str(exc),
        }
        return detail["status"], detail

    detail["status"] = "ok"
    detail["probe"] = {"status": "ok"}
    return detail["status"], detail


def check_model_config() -> Tuple[str, Dict[str, Any]]:
    detail: Dict[str, Any] = {}

    errors = []
    try:
        execution_policy = resolve_runtime_execution_policy()
        settings = execution_policy.settings
        detail.update(
            {
                "model_key": settings.model_key,
                "quantization": settings.quantization,
                "dtype": settings.dtype_name,
                "source": settings.source,
                "executionPolicy": execution_policy.to_payload(),
            }
        )
        if settings.profile_path:
            detail["profile_path"] = settings.profile_path

        spec = resolve_inference_model(settings.model_key)
        detail["model_id"] = spec.model_id
        detail["mode"] = spec.mode
    except Exception as exc:
        errors.append(str(exc))

    quantization = detail.get("quantization")
    dtype_name = detail.get("dtype")
    if quantization is not None and quantization not in {"none", "8bit", "4bit"}:
        errors.append("Unsupported quantization")
    if dtype_name is not None and dtype_name not in {"float16", "bfloat16", "float32"}:
        errors.append("Unsupported dtype")

    if errors:
        detail["status"] = "error"
        detail["errors"] = errors
    else:
        detail["status"] = "ok"

    return detail["status"], detail


def check_model_preload() -> Tuple[str, Dict[str, Any]]:
    path = get_preload_status_path()
    detail: Dict[str, Any] = {"status_path": str(path)}

    if not path.exists():
        detail["status"] = "error"
        detail["message"] = "Preload status file is missing"
        return detail["status"], detail

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        detail["status"] = "error"
        detail["message"] = f"Failed to parse preload status: {exc}"
        return detail["status"], detail

    detail.update(payload)
    if payload.get("status") == "ready":
        detail["status"] = "ok"
        return detail["status"], detail

    detail["status"] = "error"
    if "message" not in detail:
        detail["message"] = "Model preload has not completed successfully"
    return detail["status"], detail


def default_worker_name() -> str:
    return f"celery@{socket.gethostname()}"


def _check_worker_process() -> Tuple[str, Dict[str, Any]]:
    cmdline_path = Path("/proc/1/cmdline")
    detail: Dict[str, Any] = {"pid": 1}

    try:
        os.kill(1, 0)
        detail["pid_alive"] = True
    except Exception as exc:
        detail["status"] = "error"
        detail["pid_alive"] = False
        detail["message"] = str(exc)
        return detail["status"], detail

    try:
        cmdline = cmdline_path.read_text(encoding="utf-8", errors="replace").replace(
            "\x00", " "
        ).strip()
    except Exception as exc:
        detail["status"] = "error"
        detail["message"] = f"Failed to read worker process cmdline: {exc}"
        return detail["status"], detail

    detail["cmdline"] = cmdline
    if "celery" in cmdline and "worker" in cmdline:
        detail["status"] = "ok"
    else:
        detail["status"] = "error"
        detail["message"] = "PID 1 is not a Celery worker process"

    return detail["status"], detail


def check_worker_ping(
    celery_app: Celery | None = None,
    worker_name: str | None = None,
) -> Tuple[str, Dict[str, Any]]:
    broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    resolved_worker_name = worker_name or default_worker_name()
    detail: Dict[str, Any] = {
        "broker_url": broker_url,
        "worker_name": resolved_worker_name,
    }

    app = celery_app or Celery("inference_worker_healthcheck", broker=broker_url)

    try:
        replies = app.control.ping(destination=[resolved_worker_name], timeout=1.0)
    except Exception as exc:
        detail["status"] = "error"
        detail["message"] = str(exc)
        return detail["status"], detail

    if any(reply.get(resolved_worker_name) == "pong" for reply in replies):
        detail["status"] = "ok"
        return detail["status"], detail

    process_status, process_detail = _check_worker_process()
    detail["fallback_process"] = process_detail
    if process_status == "ok":
        detail["status"] = "ok"
        detail["warning"] = "Celery ping did not respond; falling back to worker process check"
    else:
        detail["status"] = "error"
        detail["message"] = "No ping response from Celery worker"

    return detail["status"], detail


def build_worker_health_payload() -> Tuple[int, Dict[str, Any]]:
    _, worker_detail = check_worker_ping()
    _, queue_detail = check_queue()
    _, storage_detail = check_storage()
    _, model_detail = check_model_config()
    _, preload_detail = check_model_preload()

    checks = {
        "worker": worker_detail,
        "queue": queue_detail,
        "storage": storage_detail,
        "model": model_detail,
        "preload": preload_detail,
    }
    summary = build_health_summary(checks)
    ready = summary["failing"] == 0
    overall_status = "ok" if ready else "degraded"
    status_code = 0 if ready else 1

    payload = {
        "status": overall_status,
        "service": "inference-worker",
        "check_type": "readiness",
        "ready": ready,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "checks": checks,
    }
    return status_code, payload
