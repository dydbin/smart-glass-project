import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

from src.models.registry import resolve_inference_model


SERVING_PROFILE_ENV = "INFERENCE_SERVING_PROFILE_PATH"
RUNTIME_ENV_VARS = {
    "model_key": "VISION_CAPTION_MODEL",
    "quantization": "VISION_CAPTION_QUANTIZATION",
    "dtype_name": "VISION_CAPTION_DTYPE",
}
PRELOAD_ENV_VARS = {
    "model_key": "VISION_PRELOAD_MODEL",
    "quantization": "VISION_PRELOAD_QUANTIZATION",
    "dtype_name": "VISION_PRELOAD_DTYPE",
}
LEGACY_DEFAULTS = {
    "model_key": "blip-base",
    "quantization": "none",
    "dtype_name": "float16",
}
TASK_SOFT_LIMIT_ENV = "VISION_TASK_SOFT_TIME_LIMIT_SEC"
TASK_HARD_LIMIT_ENV = "VISION_TASK_HARD_TIME_LIMIT_SEC"
DEFAULT_TASK_SOFT_LIMIT_SECONDS = 120
DEFAULT_TASK_HARD_LIMIT_SECONDS = 150
QWEN_FALLBACK_ENV = "VISION_QWEN_FALLBACK_MODEL"
DEFAULT_QWEN_FALLBACK_MODEL = "qwen2.5-vl-3b"


@dataclass(frozen=True)
class LoadedServingProfile:
    path: Path
    payload: Dict[str, Any]


@dataclass(frozen=True)
class ResolvedServingSettings:
    model_key: str
    quantization: str
    dtype_name: str
    source: str
    profile_path: str | None = None


@dataclass(frozen=True)
class ResolvedExecutionPolicy:
    settings: ResolvedServingSettings
    soft_time_limit_sec: int
    hard_time_limit_sec: int
    fallback_model_key: str | None = None

    def to_payload(self, *, fallback_triggered: bool = False) -> Dict[str, Any]:
        return {
            "settingsSource": self.settings.source,
            "profilePath": self.settings.profile_path,
            "selectedModelKey": self.settings.model_key,
            "softTimeLimitSec": self.soft_time_limit_sec,
            "hardTimeLimitSec": self.hard_time_limit_sec,
            "fallbackModelKey": self.fallback_model_key,
            "fallbackTriggered": fallback_triggered,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_nonempty_env(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


def _resolve_positive_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default
    try:
        parsed = int(raw_value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _resolve_profile_path(profile_path: str | None = None) -> Path | None:
    configured = (profile_path or _read_nonempty_env(SERVING_PROFILE_ENV) or "").strip()
    if not configured:
        return None
    return Path(configured)


def _normalize_default_settings(payload: Dict[str, Any]) -> Dict[str, str]:
    default = payload.get("default")
    if not isinstance(default, dict):
        raise ValueError("Serving profile is missing a 'default' section")

    normalized = {
        "model_key": str(default.get("model_key", "")).strip(),
        "quantization": str(default.get("quantization", "")).strip(),
        "dtype_name": str(default.get("dtype_name", default.get("dtype", ""))).strip(),
    }
    missing = [key for key, value in normalized.items() if not value]
    if missing:
        raise ValueError(
            "Serving profile default is missing required fields: "
            + ", ".join(sorted(missing))
        )
    return normalized


def load_serving_profile(profile_path: str | None = None) -> LoadedServingProfile | None:
    path = _resolve_profile_path(profile_path)
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Serving profile not found: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse serving profile JSON: {exc}") from exc

    _normalize_default_settings(payload)
    return LoadedServingProfile(path=path, payload=payload)


def _build_settings(
    values: Dict[str, str],
    *,
    source: str,
    profile_path: str | None = None,
) -> ResolvedServingSettings:
    return ResolvedServingSettings(
        model_key=values["model_key"],
        quantization=values["quantization"],
        dtype_name=values["dtype_name"],
        source=source,
        profile_path=profile_path,
    )


def _read_explicit_settings(env_var_map: Dict[str, str]) -> Dict[str, str | None]:
    return {
        field: _read_nonempty_env(env_name) for field, env_name in env_var_map.items()
    }


def resolve_runtime_serving_settings(
    profile_path: str | None = None,
) -> ResolvedServingSettings:
    explicit = _read_explicit_settings(RUNTIME_ENV_VARS)
    if any(value is not None for value in explicit.values()):
        return _build_settings(
            {
                "model_key": explicit["model_key"] or LEGACY_DEFAULTS["model_key"],
                "quantization": explicit["quantization"]
                or LEGACY_DEFAULTS["quantization"],
                "dtype_name": explicit["dtype_name"] or LEGACY_DEFAULTS["dtype_name"],
            },
            source="env",
        )

    loaded_profile = load_serving_profile(profile_path)
    if loaded_profile is not None:
        return _build_settings(
            _normalize_default_settings(loaded_profile.payload),
            source="profile",
            profile_path=str(loaded_profile.path),
        )

    return _build_settings(LEGACY_DEFAULTS, source="legacy_default")


def resolve_preload_serving_settings(
    profile_path: str | None = None,
) -> ResolvedServingSettings:
    base = resolve_runtime_serving_settings(profile_path)
    explicit = _read_explicit_settings(PRELOAD_ENV_VARS)
    if any(value is not None for value in explicit.values()):
        return _build_settings(
            {
                "model_key": explicit["model_key"] or base.model_key,
                "quantization": explicit["quantization"] or base.quantization,
                "dtype_name": explicit["dtype_name"] or base.dtype_name,
            },
            source="preload_env",
            profile_path=base.profile_path,
        )
    return base


def resolve_task_time_limits() -> tuple[int, int]:
    soft_limit = _resolve_positive_int_env(
        TASK_SOFT_LIMIT_ENV,
        DEFAULT_TASK_SOFT_LIMIT_SECONDS,
    )
    hard_limit = max(
        _resolve_positive_int_env(
            TASK_HARD_LIMIT_ENV,
            DEFAULT_TASK_HARD_LIMIT_SECONDS,
        ),
        soft_limit + 1,
    )
    return soft_limit, hard_limit


def _resolve_qwen_fallback_model_key(model_key: str) -> str | None:
    try:
        selected_descriptor = resolve_inference_model(model_key)
    except Exception:
        return None
    if selected_descriptor.mode != "vlm":
        return None

    fallback_model_key = (
        _read_nonempty_env(QWEN_FALLBACK_ENV) or DEFAULT_QWEN_FALLBACK_MODEL
    )
    if fallback_model_key == selected_descriptor.key:
        return None
    try:
        fallback_descriptor = resolve_inference_model(fallback_model_key)
    except Exception:
        return None
    if fallback_descriptor.mode != "vlm":
        return None
    return fallback_descriptor.key


def resolve_runtime_execution_policy(
    profile_path: str | None = None,
) -> ResolvedExecutionPolicy:
    settings = resolve_runtime_serving_settings(profile_path)
    soft_limit, hard_limit = resolve_task_time_limits()
    return ResolvedExecutionPolicy(
        settings=settings,
        soft_time_limit_sec=soft_limit,
        hard_time_limit_sec=hard_limit,
        fallback_model_key=_resolve_qwen_fallback_model_key(settings.model_key),
    )


def resolve_preload_execution_policy(
    profile_path: str | None = None,
) -> ResolvedExecutionPolicy:
    settings = resolve_preload_serving_settings(profile_path)
    soft_limit, hard_limit = resolve_task_time_limits()
    return ResolvedExecutionPolicy(
        settings=settings,
        soft_time_limit_sec=soft_limit,
        hard_time_limit_sec=hard_limit,
        fallback_model_key=_resolve_qwen_fallback_model_key(settings.model_key),
    )


def _to_float(value: object) -> float | None:
    if value in {"", None}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: object) -> int | None:
    if value in {"", None}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _build_candidate(row: Dict[str, Any], latency_budget_sec: float) -> Dict[str, Any]:
    latency_p95 = _to_float(row.get("latency_p95_sec"))
    peak_memory = _to_float(row.get("peak_memory_max_mb"))
    load_time = _to_float(row.get("load_time_sec"))
    lexical_f1 = _to_float(row.get("lexical_f1_avg"))
    images_success = _to_int(row.get("images_success")) or 0
    images_error = _to_int(row.get("images_error")) or 0

    meets_latency_budget = latency_p95 is not None and latency_p95 <= latency_budget_sec
    is_valid = images_success > 0 and images_error == 0 and latency_p95 is not None

    return {
        "model_key": str(row.get("model_key", "")).strip(),
        "model_id": str(row.get("model_id", "")).strip(),
        "quantization": str(row.get("quantization", "")).strip() or "none",
        "dtype_name": str(row.get("dtype_name", row.get("dtype", ""))).strip()
        or LEGACY_DEFAULTS["dtype_name"],
        "images_total": _to_int(row.get("images_total")) or 0,
        "images_success": images_success,
        "images_error": images_error,
        "latency_p95_sec": latency_p95,
        "peak_memory_max_mb": peak_memory,
        "load_time_sec": load_time,
        "lexical_f1_avg": lexical_f1,
        "meets_latency_budget": meets_latency_budget,
        "is_valid": is_valid,
    }


def _candidate_sort_key(candidate: Dict[str, Any]) -> tuple:
    lexical_f1 = candidate.get("lexical_f1_avg")
    return (
        0 if candidate["meets_latency_budget"] else 1,
        0 if lexical_f1 is not None else 1,
        -(lexical_f1 or 0.0),
        candidate["latency_p95_sec"] or float("inf"),
        candidate["peak_memory_max_mb"] or float("inf"),
        candidate["load_time_sec"] or float("inf"),
        candidate["model_key"],
        candidate["quantization"],
        candidate["dtype_name"],
    )


def _pick_ranked_candidates(
    results: Iterable[Dict[str, Any]], latency_budget_sec: float
) -> list[Dict[str, Any]]:
    candidates = [_build_candidate(row, latency_budget_sec) for row in results]
    ranked = [candidate for candidate in candidates if candidate["is_valid"]]
    ranked.sort(key=_candidate_sort_key)
    return ranked


def build_caption_serving_profile(
    summary_payload: Dict[str, Any],
    *,
    latency_budget_sec: float = 10.0,
    source_summary_path: str | None = None,
) -> Dict[str, Any]:
    results = summary_payload.get("results")
    if not isinstance(results, list):
        raise ValueError("Benchmark summary payload is missing a 'results' list")

    ranked_candidates = _pick_ranked_candidates(results, latency_budget_sec)
    if not ranked_candidates:
        raise ValueError("No valid caption benchmark candidates available for serving")

    default_candidate = ranked_candidates[0]
    fallback_candidate = ranked_candidates[1] if len(ranked_candidates) > 1 else None

    profile: Dict[str, Any] = {
        "profile_type": "caption_serving_profile",
        "generated_at_utc": summary_payload.get("generated_at_utc") or _utc_now(),
        "source_summary_path": source_summary_path,
        "selection_policy": {
            "latency_budget_sec": latency_budget_sec,
            "candidate_requirements": [
                "images_success > 0",
                "images_error == 0",
                "latency_p95_sec is present",
            ],
            "ranking": [
                "prefer candidates within latency budget",
                "prefer higher lexical_f1_avg when available",
                "prefer lower latency_p95_sec",
                "prefer lower peak_memory_max_mb",
                "prefer lower load_time_sec",
            ],
        },
        "default": {
            **default_candidate,
            "selection_reason": "best-ranked caption benchmark candidate",
        },
        "fallback": None,
        "candidates": ranked_candidates,
    }

    if fallback_candidate is not None:
        profile["fallback"] = {
            **fallback_candidate,
            "selection_reason": "next-best caption benchmark candidate",
        }

    return profile
