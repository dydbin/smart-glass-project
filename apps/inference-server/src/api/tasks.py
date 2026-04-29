from typing import Any

from fastapi import APIRouter, Request

from src.api.schemas import (
    VisionInferenceEnqueueRequest,
    VisionInferenceEnqueueResponse,
    VisionInferenceTaskStatusResponse,
)
from src.queue.tasks import app as celery_app
from src.queue.tasks import process_vision_inference

router = APIRouter()


def _extract_result_status(result_payload: dict[str, Any] | None) -> str | None:
    if not isinstance(result_payload, dict):
        return None

    raw_status = result_payload.get("status")
    if raw_status is None:
        return None

    normalized = str(raw_status).strip().lower()
    if normalized in {"success", "error"}:
        return normalized
    return None


def _map_task_status(
    *,
    state: str,
    ready: bool,
    celery_successful: bool,
    celery_failed: bool,
    result_status: str | None,
) -> str:
    if not ready:
        if state in {"STARTED", "RETRY", "PROGRESS"}:
            return "running"
        return "pending"

    if result_status == "success":
        return "completed"
    if result_status == "error":
        return "failed"

    if celery_failed:
        return "failed"
    if celery_successful:
        return "completed"
    return "failed"


@router.post(
    "/tasks/vision",
    response_model=VisionInferenceEnqueueResponse,
    status_code=202,
)
async def enqueue_vision_inference(
    request: Request,
    payload: VisionInferenceEnqueueRequest,
) -> VisionInferenceEnqueueResponse:
    resolved_request_id = payload.request_id or getattr(
        request.state, "request_id", None
    )
    async_result = process_vision_inference.apply_async(
        kwargs={
            "image_key": payload.image_key,
            "user_id": payload.user_id,
            "memory_id": payload.memory_id,
            "captured_at": payload.captured_at,
            "image_url": payload.image_url,
            "request_id": resolved_request_id,
            "task_type": payload.task_type,
        }
    )
    return VisionInferenceEnqueueResponse(
        taskId=async_result.id,
        state=async_result.state,
        requestId=resolved_request_id or async_result.id,
        taskType=payload.task_type,
        statusUrl=f"/tasks/{async_result.id}",
    )


@router.get(
    "/tasks/{task_id}",
    response_model=VisionInferenceTaskStatusResponse,
)
async def get_vision_inference_task(task_id: str) -> VisionInferenceTaskStatusResponse:
    async_result = celery_app.AsyncResult(task_id)
    ready = async_result.ready()
    celery_successful = async_result.successful() if ready else False
    celery_failed = async_result.failed() if ready else False

    result_payload: dict[str, Any] | None = None
    error_message: str | None = None
    if ready:
        result = async_result.result
        if isinstance(result, dict):
            result_payload = result
        elif result is not None:
            error_message = str(result)
        if celery_failed and error_message is None:
            error_message = str(async_result.result)

    result_status = _extract_result_status(result_payload)
    if result_status == "error" and error_message is None and isinstance(result_payload, dict):
        raw_message = result_payload.get("message")
        if raw_message is not None:
            error_message = str(raw_message)

    task_status = _map_task_status(
        state=async_result.state,
        ready=ready,
        celery_successful=celery_successful,
        celery_failed=celery_failed,
        result_status=result_status,
    )
    successful = task_status == "completed"

    request_id = None
    if isinstance(result_payload, dict):
        raw_request_id = result_payload.get("requestId")
        if raw_request_id is not None:
            request_id = str(raw_request_id).strip() or None

    return VisionInferenceTaskStatusResponse(
        taskId=task_id,
        state=async_result.state,
        ready=ready,
        successful=successful,
        resultStatus=result_status,
        requestId=request_id,
        result=result_payload,
        error=error_message,
        taskStatus=task_status,
    )
