import os
import io
from dataclasses import dataclass
from celery import Celery
from celery.exceptions import Retry, SoftTimeLimitExceeded
from celery.signals import worker_init
from PIL import Image, UnidentifiedImageError

from src.contracts.vlm import (
    build_request_id,
    build_vlm_error_result,
    build_vlm_success_result,
)
from src.core.logging import configure_logging, get_logger
from src.models.captioning import generate_caption
from src.models.qwen_vlm import generate_qwen_vlm_metadata
from src.models.registry import resolve_inference_model
from src.models.serving_profile import (
    resolve_runtime_execution_policy,
    resolve_task_time_limits,
)
from src.storage.s3 import (
    StorageAccessError,
    StorageConfigError,
    StorageNotFoundError,
    get_storage_service,
)
from src.worker_preload import maybe_preload_on_startup

broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
result_backend = os.getenv("CELERY_RESULT_BACKEND", broker_url)
app = Celery("inference_tasks", broker=broker_url, backend=result_backend)

TASK_SOFT_TIME_LIMIT_SECONDS, TASK_TIME_LIMIT_SECONDS = resolve_task_time_limits()


def _nonnegative_int_env(name: str, fallback: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return fallback
    try:
        return max(0, int(raw))
    except ValueError:
        return fallback


TASK_MAX_RETRIES = _nonnegative_int_env("VISION_TASK_MAX_RETRIES", 2)
TASK_RETRY_DELAY_SECONDS = _nonnegative_int_env("VISION_TASK_RETRY_DELAY_SEC", 10)

app.conf.update(
    task_track_started=True,
    result_expires=3600,
    task_soft_time_limit=TASK_SOFT_TIME_LIMIT_SECONDS,
    task_time_limit=TASK_TIME_LIMIT_SECONDS,
)
configure_logging()
logger = get_logger(__name__)


@dataclass(frozen=True)
class InferenceFailurePolicy:
    error_code: str
    retryable: bool
    category: str


@worker_init.connect
def _preload_worker_model_on_startup(**_: object) -> None:
    maybe_preload_on_startup()


def _should_retry_with_qwen_fallback(error: Exception) -> bool:
    message = str(error).lower()
    retryable_markers = (
        "out of memory",
        "cuda error",
        "cublas",
        "device-side assert",
        "allocation",
    )
    return any(marker in message for marker in retryable_markers)


def _classify_inference_error(error: Exception) -> InferenceFailurePolicy:
    if isinstance(error, (SoftTimeLimitExceeded, TimeoutError)):
        return InferenceFailurePolicy("inference_timeout", True, "timeout")
    if isinstance(error, StorageNotFoundError):
        return InferenceFailurePolicy("source_image_not_found", False, "source_image")
    if isinstance(error, StorageConfigError):
        return InferenceFailurePolicy("storage_config_error", False, "storage")
    if isinstance(error, StorageAccessError):
        return InferenceFailurePolicy("storage_access_error", True, "storage")
    if isinstance(error, UnidentifiedImageError):
        return InferenceFailurePolicy("invalid_source_image", False, "input")
    if isinstance(error, ValueError):
        return InferenceFailurePolicy("invalid_inference_request", False, "input")
    if isinstance(error, RuntimeError):
        return InferenceFailurePolicy("model_runtime_error", True, "model")
    return InferenceFailurePolicy("inference_task_error", True, "unknown")


def _build_error_details(
    *,
    error: Exception,
    failure_policy: InferenceFailurePolicy,
) -> dict[str, object]:
    return {
        "category": failure_policy.category,
        "reason": failure_policy.error_code,
        "exceptionType": type(error).__name__,
        "retryable": failure_policy.retryable,
        "source": "inference_worker",
        "taskTimeLimit": {
            "softSec": TASK_SOFT_TIME_LIMIT_SECONDS,
            "hardSec": TASK_TIME_LIMIT_SECONDS,
        },
    }


def _task_called_directly(task_request: object | None) -> bool:
    return bool(getattr(task_request, "called_directly", False))


def _task_retry_count(task_request: object | None) -> int:
    try:
        return max(0, int(getattr(task_request, "retries", 0) or 0))
    except (TypeError, ValueError):
        return 0


def _should_retry_task(
    *,
    retryable: bool,
    task_request: object | None,
    max_retries: int = TASK_MAX_RETRIES,
) -> bool:
    if not retryable or max_retries <= 0:
        return False
    if _task_called_directly(task_request):
        return False
    return _task_retry_count(task_request) < max_retries


@app.task(
    bind=True,
    name="process_vision_inference",
    soft_time_limit=TASK_SOFT_TIME_LIMIT_SECONDS,
    time_limit=TASK_TIME_LIMIT_SECONDS,
    max_retries=TASK_MAX_RETRIES,
    default_retry_delay=TASK_RETRY_DELAY_SECONDS,
)
def process_vision_inference(
    self,
    image_key,
    user_id,
    memory_id=None,
    captured_at=None,
    image_url=None,
    request_id=None,
    task_type="caption",
):
    model_key = "blip-base"
    quantization = "none"
    dtype_name = "float16"
    settings_source = "legacy_default"
    execution_policy_payload = None
    resolved_request_id = build_request_id(request_id)
    content_type = None
    model_mode = "unknown"
    fallback_triggered = False

    try:
        execution_policy = resolve_runtime_execution_policy()
        settings = execution_policy.settings
        model_key = settings.model_key
        quantization = settings.quantization
        dtype_name = settings.dtype_name
        settings_source = settings.source
        execution_policy_payload = execution_policy.to_payload()
        model_descriptor = resolve_inference_model(model_key)
        model_mode = model_descriptor.mode
        storage_object = get_storage_service().read_object(image_key)
        content_type = storage_object.content_type
        image_data = storage_object.body
        raw_image = Image.open(io.BytesIO(image_data)).convert("RGB")
        if model_descriptor.mode == "vlm":
            try:
                result = generate_qwen_vlm_metadata(
                    image=raw_image,
                    model_key=model_key,
                    quantization=quantization,
                    dtype_name=dtype_name,
                )
            except Exception as primary_error:
                fallback_model_key = None
                if _should_retry_with_qwen_fallback(primary_error):
                    fallback_model_key = execution_policy.fallback_model_key
                if fallback_model_key is None:
                    raise

                logger.warning(
                    "Primary VLM inference failed, retrying with fallback model",
                    extra={
                        "task_name": "process_vision_inference",
                        "request_id": resolved_request_id,
                        "image_key": image_key,
                        "user_id": user_id,
                        "model_key": model_key,
                        "fallback_model_key": fallback_model_key,
                        "settings_source": settings_source,
                        "error_code": "vlm_primary_model_failed",
                    },
                )
                result = generate_qwen_vlm_metadata(
                    image=raw_image,
                    model_key=fallback_model_key,
                    quantization=quantization,
                    dtype_name=dtype_name,
                )
                model_key = fallback_model_key
                model_descriptor = resolve_inference_model(model_key)
                fallback_triggered = True
                execution_policy_payload = execution_policy.to_payload(
                    fallback_triggered=True
                )
            metadata = result["metadata"]
            pipeline_output = result.get("pipeline_output")
        else:
            result = generate_caption(
                image=raw_image,
                model_key=model_key,
                quantization=quantization,
                dtype_name=dtype_name,
            )
            metadata = None
            pipeline_output = None

        logger.info(
            "Inference task completed",
            extra={
                "task_name": "process_vision_inference",
                "request_id": resolved_request_id,
                "image_key": image_key,
                "memory_id": memory_id,
                "user_id": user_id,
                "model_key": model_key,
                "model_mode": model_mode,
                "quantization": quantization,
                "settings_source": settings_source,
                "soft_time_limit_sec": execution_policy.soft_time_limit_sec,
                "hard_time_limit_sec": execution_policy.hard_time_limit_sec,
                "fallback_model_key": execution_policy.fallback_model_key,
                "fallback_triggered": fallback_triggered,
                "latency_sec": round(result["elapsed_sec"], 4),
                "peak_memory_mb": result["peak_memory_mb"],
            },
        )

        return build_vlm_success_result(
            request_id=resolved_request_id,
            user_id=user_id,
            image_key=image_key,
            model_key=model_key,
            quantization=quantization,
            dtype_name=dtype_name,
            generation_result=result,
            memory_id=memory_id,
            image_url=image_url,
            captured_at=captured_at,
            task_type=task_type,
            content_type=content_type,
            inference_metadata=metadata,
            pipeline_output=pipeline_output,
            execution_policy=execution_policy_payload,
        )
    except Retry:
        raise
    except Exception as e:
        failure_policy = _classify_inference_error(e)
        error_details = _build_error_details(
            error=e,
            failure_policy=failure_policy,
        )
        task_request = getattr(self, "request", None)
        retry_count = _task_retry_count(task_request)
        
        if _should_retry_task(retryable=failure_policy.retryable, task_request=task_request):
            logger.warning(
                "Inference task failed with retryable error, scheduling retry",
                extra={
                    "task_name": "process_vision_inference",
                    "request_id": resolved_request_id,
                    "image_key": image_key,
                    "memory_id": memory_id,
                    "user_id": user_id,
                    "model_key": model_key,
                    "model_mode": model_mode,
                    "quantization": quantization,
                    "settings_source": settings_source,
                    "error_code": failure_policy.error_code,
                    "retry_count": retry_count,
                    "max_retries": TASK_MAX_RETRIES,
                    "retry_delay_sec": TASK_RETRY_DELAY_SECONDS,
                },
            )
            raise self.retry(exc=e, countdown=TASK_RETRY_DELAY_SECONDS)

        logger.exception(
            "Inference task failed",
            extra={
                "task_name": "process_vision_inference",
                "request_id": resolved_request_id,
                "image_key": image_key,
                "memory_id": memory_id,
                "user_id": user_id,
                "model_key": model_key,
                "model_mode": model_mode,
                "quantization": quantization,
                "settings_source": settings_source,
                "soft_time_limit_sec": TASK_SOFT_TIME_LIMIT_SECONDS,
                "hard_time_limit_sec": TASK_TIME_LIMIT_SECONDS,
                "error_code": failure_policy.error_code,
                "retryable": failure_policy.retryable,
                "failure_category": failure_policy.category,
                "retry_count": retry_count,
                "max_retries": TASK_MAX_RETRIES,
            },
        )
        return build_vlm_error_result(
            request_id=resolved_request_id,
            user_id=user_id,
            image_key=image_key,
            error=e,
            model_key=model_key,
            quantization=quantization,
            dtype_name=dtype_name,
            memory_id=memory_id,
            image_url=image_url,
            captured_at=captured_at,
            task_type=task_type,
            content_type=content_type,
            error_code=failure_policy.error_code,
            retryable=failure_policy.retryable,
            execution_policy=execution_policy_payload,
            error_details=error_details,
        )
