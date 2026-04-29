from __future__ import annotations

import unittest

from src.api.intake import build_capture_upload_response, build_worker_task_kwargs
from src.api.pipeline import (
    CapturePipelineError,
    CapturePipelineSettings,
    CaptureProcessingPipeline,
    CaptureTaskOutcome,
)
from src.api.schemas import CaptureUploadRequest
from src.database.memory_store import MemoryStoreOutcome


class RecordingTaskTransport:
    def __init__(self, outcome: CaptureTaskOutcome) -> None:
        self.outcome = outcome
        self.last_kwargs: dict[str, object] | None = None
        self.last_timeout_sec: float | None = None

    def submit_and_wait(
        self,
        *,
        task_kwargs: dict[str, object],
        timeout_sec: float,
    ) -> CaptureTaskOutcome:
        self.last_kwargs = task_kwargs
        self.last_timeout_sec = timeout_sec
        return self.outcome


class RecordingMemoryStoreClient:
    backend_name = "postgres"

    def __init__(self, outcome: MemoryStoreOutcome) -> None:
        self.outcome = outcome
        self.last_worker_result: dict[str, object] | None = None

    def persist_vlm_result(self, worker_result: dict[str, object]) -> MemoryStoreOutcome:
        self.last_worker_result = worker_result
        return self.outcome


class FailingMemoryStoreClient:
    backend_name = "postgres"

    def __init__(self, message: str) -> None:
        self.message = message
        self.last_worker_result: dict[str, object] | None = None

    def persist_vlm_result(self, worker_result: dict[str, object]) -> MemoryStoreOutcome:
        self.last_worker_result = worker_result
        raise RuntimeError(self.message)


class CaptureProcessingPipelineTests(unittest.TestCase):
    def _build_payload(self) -> CaptureUploadRequest:
        return CaptureUploadRequest(
            captureId="capture-10",
            requestId="req-10",
            memoryId="mem-10",
            userId="user-10",
            taskType="metadata",
            capturedAt="2026-04-07T08:00:00Z",
            fileName="glass-photo.jpg",
            imageUrl="https://example.com/captures/glass-photo.jpg",
        )

    def _build_worker_result(
        self, payload: CaptureUploadRequest
    ) -> tuple[dict[str, object], dict[str, object]]:
        capture_response = build_capture_upload_response(payload)
        expected_kwargs = build_worker_task_kwargs(capture_response)
        worker_result: dict[str, object] = {
            "status": "success",
            "requestId": capture_response.requestId,
            "taskType": capture_response.taskType,
            "memoryId": capture_response.memoryId,
            "userId": capture_response.userId,
            "capturedAt": capture_response.capturedAt,
            "sourceImage": {
                "imageKey": capture_response.sourceImage.imageKey,
                "imageUrl": capture_response.sourceImage.imageUrl,
                "contentType": capture_response.sourceImage.contentType,
            },
            "metadata": {
                "caption": "a laptop is on the table",
                "sceneSummary": "tabletop scene",
                "detectedObjects": ["laptop"],
                "tags": ["desk"],
                "ocrText": "notes",
                "positionHint": "on the table",
                "location": {"name": "office"},
            },
            "pipelineOutput": {
                "scene_summary": "tabletop scene",
                "location_context": "office desk",
                "objects": [],
            },
        }
        return worker_result, expected_kwargs

    def test_pipeline_dispatches_worker_and_skips_memory_storage_when_disabled(self) -> None:
        payload = self._build_payload()
        worker_result, expected_kwargs = self._build_worker_result(payload)
        task_transport = RecordingTaskTransport(
            CaptureTaskOutcome(task_id="task-123", result=worker_result)
        )
        pipeline = CaptureProcessingPipeline(
            settings=CapturePipelineSettings(
                broker_url="redis://redis:6379/0",
                result_backend_url="redis://redis:6379/1",
                enable_memory_store=False,
                worker_timeout_sec=15.0,
            ),
            task_transport=task_transport,
            memory_store_client=None,
        )

        response = pipeline.process(payload)

        self.assertEqual(task_transport.last_kwargs, expected_kwargs)
        self.assertEqual(task_transport.last_timeout_sec, 15.0)
        self.assertEqual(response.status, "completed")
        self.assertEqual(response.capture.captureId, "capture-10")
        self.assertEqual(response.worker.taskId, "task-123")
        self.assertEqual(response.worker.status, "success")
        self.assertEqual(response.memoryStore.status, "skipped")
        self.assertEqual(response.memoryStore.backend, "disabled")

    def test_pipeline_dispatches_worker_and_persists_memory_when_enabled(self) -> None:
        payload = self._build_payload()
        worker_result, expected_kwargs = self._build_worker_result(payload)
        task_transport = RecordingTaskTransport(
            CaptureTaskOutcome(task_id="task-123", result=worker_result)
        )
        memory_store = RecordingMemoryStoreClient(
            MemoryStoreOutcome(stored_count=1, total_user_memories={"user-10": 1})
        )
        pipeline = CaptureProcessingPipeline(
            settings=CapturePipelineSettings(
                broker_url="redis://redis:6379/0",
                result_backend_url="redis://redis:6379/1",
                enable_memory_store=True,
                database_url="postgresql://postgres:postgres@postgres:5432/smart_glass",
                worker_timeout_sec=15.0,
            ),
            task_transport=task_transport,
            memory_store_client=memory_store,
        )

        response = pipeline.process(payload)

        self.assertEqual(task_transport.last_kwargs, expected_kwargs)
        self.assertEqual(task_transport.last_timeout_sec, 15.0)
        self.assertEqual(memory_store.last_worker_result, worker_result)
        self.assertEqual(response.status, "completed")
        self.assertEqual(response.capture.captureId, "capture-10")
        self.assertEqual(response.worker.taskId, "task-123")
        self.assertEqual(response.worker.status, "success")
        self.assertEqual(response.memoryStore.backend, "postgres")
        self.assertEqual(response.memoryStore.storedCount, 1)
        self.assertEqual(response.memoryStore.totalUserMemories["user-10"], 1)

    def test_pipeline_returns_partial_when_memory_storage_fails(self) -> None:
        payload = self._build_payload()
        worker_result, _ = self._build_worker_result(payload)
        task_transport = RecordingTaskTransport(
            CaptureTaskOutcome(task_id="task-123", result=worker_result)
        )
        memory_store = FailingMemoryStoreClient("database timeout")
        pipeline = CaptureProcessingPipeline(
            settings=CapturePipelineSettings(
                broker_url="redis://redis:6379/0",
                result_backend_url="redis://redis:6379/1",
                enable_memory_store=True,
                database_url="postgresql://postgres:postgres@postgres:5432/smart_glass",
                worker_timeout_sec=15.0,
            ),
            task_transport=task_transport,
            memory_store_client=memory_store,
        )

        response = pipeline.process(payload)

        self.assertEqual(response.status, "partial")
        self.assertEqual(response.worker.status, "success")
        self.assertEqual(response.memoryStore.status, "error")
        self.assertEqual(
            response.memoryStore.error,
            "memory store persistence failed: database timeout",
        )

    def test_pipeline_requires_database_url_when_memory_storage_is_enabled(self) -> None:
        with self.assertRaisesRegex(
            CapturePipelineError,
            "Memory storage is enabled but API_CAPTURE_DATABASE_URL is not configured",
        ):
            CaptureProcessingPipeline.from_settings(
                CapturePipelineSettings(
                    broker_url="redis://redis:6379/0",
                    result_backend_url="redis://redis:6379/1",
                    enable_memory_store=True,
                )
            )


if __name__ == "__main__":
    unittest.main()
