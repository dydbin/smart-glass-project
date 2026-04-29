import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from src.api.main import app


class ApiTaskRoutesTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_enqueue_vision_inference_returns_task_id(self) -> None:
        fake_async_result = MagicMock()
        fake_async_result.id = "task-123"
        fake_async_result.state = "PENDING"

        with patch(
            "src.api.tasks.process_vision_inference.apply_async",
            return_value=fake_async_result,
        ) as mocked_apply_async:
            response = self.client.post(
                "/tasks/vision",
                json={
                    "imageKey": "captures/test.png",
                    "userId": "user-1",
                    "requestId": "req-1",
                    "taskType": "metadata",
                },
            )

        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["taskId"], "task-123")
        self.assertEqual(response.json()["requestId"], "req-1")
        self.assertEqual(response.json()["taskType"], "metadata")
        self.assertEqual(response.json()["statusUrl"], "/tasks/task-123")
        self.assertEqual(
            mocked_apply_async.call_args.kwargs["kwargs"]["request_id"],
            "req-1",
        )

    def test_enqueue_vision_inference_uses_middleware_request_id_when_payload_omits_it(
        self,
    ) -> None:
        fake_async_result = MagicMock()
        fake_async_result.id = "task-456"
        fake_async_result.state = "PENDING"

        with patch(
            "src.api.tasks.process_vision_inference.apply_async",
            return_value=fake_async_result,
        ) as mocked_apply_async:
            response = self.client.post(
                "/tasks/vision",
                json={
                    "imageKey": "captures/test.png",
                    "userId": "user-1",
                    "taskType": "metadata",
                },
                headers={"x-request-id": "req-from-header"},
            )

        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["taskId"], "task-456")
        self.assertEqual(response.json()["requestId"], "req-from-header")
        self.assertEqual(response.json()["taskType"], "metadata")
        self.assertEqual(response.json()["statusUrl"], "/tasks/task-456")
        self.assertEqual(
            mocked_apply_async.call_args.kwargs["kwargs"]["request_id"],
            "req-from-header",
        )

    def test_enqueue_vision_inference_rejects_unknown_task_type(self) -> None:
        response = self.client.post(
            "/tasks/vision",
            json={
                "imageKey": "captures/test.png",
                "userId": "user-1",
                "taskType": "thumbnail",
            },
        )

        self.assertEqual(response.status_code, 422)

    def test_get_task_returns_completed_payload(self) -> None:
        fake_async_result = MagicMock()
        fake_async_result.state = "SUCCESS"
        fake_async_result.ready.return_value = True
        fake_async_result.successful.return_value = True
        fake_async_result.failed.return_value = False
        fake_async_result.result = {"status": "success", "requestId": "req-1"}

        with patch(
            "src.api.tasks.celery_app.AsyncResult",
            return_value=fake_async_result,
        ):
            response = self.client.get("/tasks/task-123")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["taskStatus"], "completed")
        self.assertEqual(response.json()["resultStatus"], "success")
        self.assertTrue(response.json()["successful"])
        self.assertEqual(response.json()["requestId"], "req-1")
        self.assertEqual(response.json()["result"]["status"], "success")

    def test_get_task_omits_top_level_request_id_when_result_is_not_ready(self) -> None:
        fake_async_result = MagicMock()
        fake_async_result.state = "PENDING"
        fake_async_result.ready.return_value = False

        with patch(
            "src.api.tasks.celery_app.AsyncResult",
            return_value=fake_async_result,
        ):
            response = self.client.get("/tasks/task-789")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["taskStatus"], "pending")
        self.assertIsNone(response.json()["resultStatus"])
        self.assertIsNone(response.json()["requestId"])

    def test_get_task_marks_error_payload_as_failed_even_if_celery_state_is_success(
        self,
    ) -> None:
        fake_async_result = MagicMock()
        fake_async_result.state = "SUCCESS"
        fake_async_result.ready.return_value = True
        fake_async_result.successful.return_value = True
        fake_async_result.failed.return_value = False
        fake_async_result.result = {
            "status": "error",
            "requestId": "req-err-1",
            "message": "model load failed",
            "errorCode": "inference_task_error",
        }

        with patch(
            "src.api.tasks.celery_app.AsyncResult",
            return_value=fake_async_result,
        ):
            response = self.client.get("/tasks/task-error")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["taskStatus"], "failed")
        self.assertEqual(response.json()["resultStatus"], "error")
        self.assertFalse(response.json()["successful"])
        self.assertEqual(response.json()["requestId"], "req-err-1")
        self.assertEqual(response.json()["error"], "model load failed")


if __name__ == "__main__":
    unittest.main()
