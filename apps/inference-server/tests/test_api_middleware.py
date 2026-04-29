import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.main import app


class MiddlewareTestCase(unittest.TestCase):
    def setUp(self) -> None:
        if not any(route.path == "/_test/error" for route in app.router.routes):
            app.add_api_route("/_test/error", self._raise_error, methods=["GET"])
        self.client = TestClient(app)

    @staticmethod
    async def _raise_error() -> None:
        raise RuntimeError("Synthetic error for middleware verification")

    def test_liveness_returns_request_id(self) -> None:
        response = self.client.get("/health/live")

        self.assertEqual(response.status_code, 200)
        self.assertIn("x-request-id", response.headers)
        self.assertEqual(response.json()["status"], "ok")
        self.assertEqual(response.json()["check_type"], "liveness")
        self.assertEqual(response.json()["service"], "inference-server")
        self.assertEqual(response.json()["summary"]["total"], 1)
        self.assertEqual(response.json()["summary"]["passing"], 1)
        self.assertEqual(response.json()["summary"]["failing"], 0)

    def test_readiness_returns_request_id(self) -> None:
        with patch("src.api.health._check_queue", return_value=("ok", {"status": "ok"})):
            with patch(
                "src.api.health._check_storage",
                return_value=("ok", {"status": "ok"}),
            ):
                with patch(
                    "src.api.health._check_model_config",
                    return_value=("ok", {"status": "ok"}),
                ):
                    response = self.client.get("/health/ready")

        self.assertEqual(response.status_code, 200)
        self.assertIn("x-request-id", response.headers)
        self.assertEqual(response.json()["status"], "ok")
        self.assertEqual(response.json()["check_type"], "readiness")
        self.assertEqual(response.json()["service"], "inference-server")
        self.assertTrue(response.json()["ready"])
        self.assertEqual(response.json()["summary"]["total"], 4)
        self.assertEqual(response.json()["summary"]["passing"], 4)
        self.assertEqual(response.json()["summary"]["failing"], 0)
        self.assertEqual(response.json()["summary"]["failingChecks"], [])

    def test_readiness_returns_503_when_dependency_is_unhealthy(self) -> None:
        with patch(
            "src.api.health._check_queue",
            return_value=("error", {"status": "error", "message": "redis down"}),
        ):
            with patch(
                "src.api.health._check_storage",
                return_value=("ok", {"status": "ok"}),
            ):
                with patch(
                    "src.api.health._check_model_config",
                    return_value=("ok", {"status": "ok"}),
                ):
                    response = self.client.get("/health/ready")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["status"], "degraded")
        self.assertEqual(response.json()["check_type"], "readiness")
        self.assertEqual(response.json()["checks"]["queue"]["status"], "error")
        self.assertFalse(response.json()["ready"])
        self.assertEqual(response.json()["summary"]["total"], 4)
        self.assertEqual(response.json()["summary"]["passing"], 3)
        self.assertEqual(response.json()["summary"]["failing"], 1)
        self.assertEqual(response.json()["summary"]["failingChecks"], ["queue"])
        self.assertEqual(response.json()["summary"]["statuses"]["queue"], "error")

    def test_health_alias_points_to_readiness(self) -> None:
        with patch("src.api.health._check_queue", return_value=("ok", {"status": "ok"})):
            with patch(
                "src.api.health._check_storage",
                return_value=("ok", {"status": "ok"}),
            ):
                with patch(
                    "src.api.health._check_model_config",
                    return_value=("ok", {"status": "ok"}),
                ):
                    response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["check_type"], "readiness")

    def test_http_exception_is_wrapped(self) -> None:
        response = self.client.get(
            "/missing-route", headers={"x-request-id": "req-http-1"}
        )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"]["code"], "http_error")
        self.assertEqual(response.json()["error"]["request_id"], "req-http-1")

    def test_unhandled_exception_is_wrapped(self) -> None:
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/_test/error", headers={"x-request-id": "req-500-1"})

        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            response.json()["error"]["code"],
            "internal_server_error",
        )
        self.assertEqual(response.json()["error"]["request_id"], "req-500-1")


if __name__ == "__main__":
    unittest.main()
