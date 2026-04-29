import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from src.core.logging import get_logger
from src.models.captioning import (
    DEFAULT_DEVICE as DEFAULT_CAPTION_DEVICE,
    get_caption_model_components,
)
from src.models.qwen_vlm import (
    DEFAULT_DEVICE as DEFAULT_QWEN_DEVICE,
    get_qwen_vlm_components,
)
from src.models.registry import resolve_inference_model
from src.models.serving_profile import resolve_preload_execution_policy

logger = get_logger(__name__)


def get_preload_status_path() -> Path:
    raw_path = (
        os.getenv("INFERENCE_PRELOAD_STATUS_PATH", "").strip()
        or "/tmp/inference-worker-preload.json"
    )
    return Path(raw_path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _configured_model_settings() -> Dict[str, Any]:
    resolved = resolve_preload_execution_policy()
    return {
        "model_key": resolved.settings.model_key,
        "quantization": resolved.settings.quantization,
        "dtype_name": resolved.settings.dtype_name,
        "source": resolved.settings.source,
        "profile_path": resolved.settings.profile_path or "",
        "execution_policy": resolved.to_payload(),
    }


def _write_preload_state(payload: Dict[str, Any]) -> None:
    path = get_preload_status_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")


def clear_preload_state() -> None:
    path = get_preload_status_path()
    if path.exists():
        path.unlink()


def preload_configured_model() -> Dict[str, Any]:
    loading_payload: Dict[str, Any] = {
        "service": "inference-worker",
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "started_at": _utc_now(),
    }
    settings: Dict[str, Any] = {
        "model_key": "",
        "quantization": "",
        "dtype_name": "",
        "source": "unknown",
        "profile_path": "",
        "execution_policy": {
            "settingsSource": "unknown",
            "profilePath": None,
            "selectedModelKey": None,
            "softTimeLimitSec": None,
            "hardTimeLimitSec": None,
            "fallbackModelKey": None,
            "fallbackTriggered": False,
        },
    }
    model_key = "unknown"
    model_mode = "unknown"

    try:
        settings = _configured_model_settings()
        descriptor = resolve_inference_model(settings["model_key"])
        model_key = descriptor.key
        model_mode = descriptor.mode
        device_name = (
            DEFAULT_QWEN_DEVICE if descriptor.mode == "vlm" else DEFAULT_CAPTION_DEVICE
        )
        loading_payload = {
            **loading_payload,
            "status": "loading",
            "model_key": descriptor.key,
            "model_id": descriptor.model_id,
            "model_mode": descriptor.mode,
            "quantization": settings["quantization"],
            "dtype_name": settings["dtype_name"],
            "settings_source": settings["source"],
            "profile_path": settings["profile_path"] or None,
            "executionPolicy": settings["execution_policy"],
            "device": device_name,
        }
        _write_preload_state(loading_payload)

        if descriptor.mode == "vlm":
            _, _, model, _ = get_qwen_vlm_components(
                model_key=descriptor.key,
                quantization=settings["quantization"],
                device=device_name,
                dtype_name=settings["dtype_name"],
            )
        else:
            _, _, model, _ = get_caption_model_components(
                model_key=descriptor.key,
                quantization=settings["quantization"],
                device=device_name,
                dtype_name=settings["dtype_name"],
            )
    except Exception as exc:
        error_payload = {
            **loading_payload,
            "status": "error",
            "error": str(exc),
            "failed_at": _utc_now(),
        }
        _write_preload_state(error_payload)
        logger.exception(
            "Worker model preload failed",
            extra={
                "task_name": "worker_preload",
                "model_key": model_key,
                "model_mode": model_mode,
                "quantization": settings["quantization"],
                "dtype_name": settings["dtype_name"],
                "settings_source": settings["source"],
                "error_code": "worker_preload_failed",
            },
        )
        raise

    ready_payload = {
        **loading_payload,
        "status": "ready",
        "ready_at": _utc_now(),
        "load_time_sec": round(float(getattr(model, "_load_time_sec", 0.0)), 4),
    }
    _write_preload_state(ready_payload)
    logger.info(
        "Worker model preload completed",
        extra={
            "task_name": "worker_preload",
            "model_key": model_key,
            "model_mode": model_mode,
            "quantization": settings["quantization"],
            "dtype_name": settings["dtype_name"],
            "settings_source": settings["source"],
            "load_time_sec": ready_payload["load_time_sec"],
        },
    )
    return ready_payload


def maybe_preload_on_startup() -> None:
    enabled = os.getenv("INFERENCE_WORKER_PRELOAD_ON_STARTUP", "0").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return

    clear_preload_state()
    try:
        preload_configured_model()
    except Exception:
        fail_fast = os.getenv("INFERENCE_PRELOAD_FAIL_FAST", "1").strip().lower()
        if fail_fast in {"1", "true", "yes", "on"}:
            raise
