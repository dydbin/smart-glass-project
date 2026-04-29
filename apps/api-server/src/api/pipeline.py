from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol

from src.api.intake import build_capture_upload_response, build_worker_task_kwargs
from src.api.schemas import (
    CaptureMemoryStoreExecutionPayload,
    CaptureProcessingResponse,
    CaptureUploadRequest,
    CaptureWorkerExecutionPayload,
)
from src.database.memory_store import MemoryStoreOutcome, PostgresMemoryStoreClient


@dataclass(frozen=True, slots=True)
class CapturePipelineSettings:
    broker_url: str
    result_backend_url: str
    enable_memory_store: bool = False
    database_url: str | None = None
    worker_timeout_sec: float = 180.0
    worker_task_name: str = "process_vision_inference"


@dataclass(frozen=True, slots=True)
class CaptureTaskOutcome:
    task_id: str | None
    result: dict[str, Any]


class CapturePipelineError(RuntimeError):
    pass


class TaskTransport(Protocol):
    def submit_and_wait(
        self,
        *,
        task_kwargs: dict[str, Any],
        timeout_sec: float,
    ) -> CaptureTaskOutcome: ...

    def check_health(self) -> None: ...


class MemoryStoreClient(Protocol):
    backend_name: str

    def persist_vlm_result(
        self, worker_result: dict[str, Any]
    ) -> MemoryStoreOutcome: ...

    def check_health(self) -> None: ...


class CeleryTaskTransport:
    def __init__(
        self,
        *,
        broker_url: str,
        result_backend_url: str,
        task_name: str,
    ) -> None:
        self._broker_url = broker_url
        self._result_backend_url = result_backend_url
        self._task_name = task_name
        self._celery_app = None

    def _get_celery_app(self):
        if self._celery_app is not None:
            return self._celery_app
        try:
            from celery import Celery
        except ImportError as exc:  # pragma: no cover - exercised in Docker
            raise CapturePipelineError(
                "celery is required to dispatch inference tasks"
            ) from exc

        self._celery_app = Celery(
            "api_server_capture_pipeline",
            broker=self._broker_url,
            backend=self._result_backend_url,
        )
        return self._celery_app

    def submit_and_wait(
        self,
        *,
        task_kwargs: dict[str, Any],
        timeout_sec: float,
    ) -> CaptureTaskOutcome:
        celery_app = self._get_celery_app()
        async_result = celery_app.send_task(self._task_name, kwargs=task_kwargs)
        try:
            result = async_result.get(timeout=timeout_sec, propagate=True)
        except Exception as exc:  # pragma: no cover - network/runtime failure
            raise CapturePipelineError(
                f"Inference worker dispatch failed: {exc}"
            ) from exc
        if not isinstance(result, dict):
            raise CapturePipelineError("Inference worker returned an invalid payload")
        return CaptureTaskOutcome(task_id=async_result.id, result=result)

    def check_health(self) -> None:
        celery_app = self._get_celery_app()
        try:
            with celery_app.connection_for_read() as connection:
                connection.ensure_connection(max_retries=0)
        except Exception as exc:
            raise CapturePipelineError(f"Celery broker is unavailable: {exc}") from exc


def _normalize_status(value: Any) -> str:
    return " ".join(str(value).split()).strip().lower()


def _default_float(name: str, fallback: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return fallback
    try:
        return float(raw)
    except ValueError:
        return fallback


def _default_bool(name: str, fallback: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return fallback
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return fallback


def build_default_capture_pipeline() -> "CaptureProcessingPipeline":
    database_url = os.getenv("API_CAPTURE_DATABASE_URL", "").strip() or None
    settings = CapturePipelineSettings(
        broker_url=os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0").strip()
        or "redis://redis:6379/0",
        result_backend_url=os.getenv(
            "CELERY_RESULT_BACKEND", "redis://redis:6379/1"
        ).strip()
        or "redis://redis:6379/1",
        enable_memory_store=_default_bool(
            "API_CAPTURE_ENABLE_MEMORY_STORE",
            bool(database_url),
        ),
        database_url=database_url,
        worker_timeout_sec=_default_float("API_CAPTURE_WORKER_TIMEOUT_SEC", 180.0),
        worker_task_name=os.getenv(
            "API_CAPTURE_WORKER_TASK_NAME", "process_vision_inference"
        ).strip()
        or "process_vision_inference",
    )
    return CaptureProcessingPipeline.from_settings(settings)


class CaptureProcessingPipeline:
    def __init__(
        self,
        *,
        settings: CapturePipelineSettings,
        task_transport: TaskTransport,
        memory_store_client: MemoryStoreClient | None = None,
    ) -> None:
        self.settings = settings
        self.task_transport = task_transport
        self.memory_store_client = memory_store_client

    @classmethod
    def from_settings(cls, settings: CapturePipelineSettings) -> "CaptureProcessingPipeline":
        task_transport = CeleryTaskTransport(
            broker_url=settings.broker_url,
            result_backend_url=settings.result_backend_url,
            task_name=settings.worker_task_name,
        )
        memory_store_client = None
        if settings.enable_memory_store:
            if not settings.database_url:
                raise CapturePipelineError(
                    "Memory storage is enabled but API_CAPTURE_DATABASE_URL is not configured"
                )
            memory_store_client = PostgresMemoryStoreClient(settings.database_url)
        return cls(
            settings=settings,
            task_transport=task_transport,
            memory_store_client=memory_store_client,
        )

    def process(self, payload: CaptureUploadRequest) -> CaptureProcessingResponse:
        capture_response = build_capture_upload_response(payload)
        task_kwargs = build_worker_task_kwargs(capture_response)
        task_outcome = self.task_transport.submit_and_wait(
            task_kwargs=task_kwargs,
            timeout_sec=self.settings.worker_timeout_sec,
        )

        worker_result = task_outcome.result
        worker_status = _normalize_status(worker_result.get("status"))
        if worker_status != "success":
            message = worker_result.get("message") or "Inference worker returned an error"
            raise CapturePipelineError(str(message))

        memory_store_payload = CaptureMemoryStoreExecutionPayload(
            backend=(
                self.memory_store_client.backend_name
                if self.memory_store_client is not None
                else "disabled"
            ),
            status="skipped",
            storedCount=None,
            totalUserMemories={},
            error=None,
        )
        response_status = "completed"

        if self.memory_store_client is not None:
            try:
                store_outcome = self.memory_store_client.persist_vlm_result(worker_result)
                memory_store_payload = CaptureMemoryStoreExecutionPayload(
                    backend=self.memory_store_client.backend_name,
                    status="success",
                    storedCount=store_outcome.stored_count,
                    totalUserMemories=store_outcome.total_user_memories,
                    error=None,
                )
            except Exception as exc:
                response_status = "partial"
                memory_store_payload = CaptureMemoryStoreExecutionPayload(
                    backend=self.memory_store_client.backend_name,
                    status="error",
                    storedCount=None,
                    totalUserMemories={},
                    error=f"memory store persistence failed: {exc}",
                )

        return CaptureProcessingResponse(
            status=response_status,
            capture=capture_response,
            worker=CaptureWorkerExecutionPayload(
                taskId=task_outcome.task_id,
                status="success",
                result=worker_result,
                error=None,
            ),
            memoryStore=memory_store_payload,
        )

    def check_health(self) -> None:
        self.task_transport.check_health()
        if self.memory_store_client is not None:
            self.memory_store_client.check_health()
