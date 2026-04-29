import unittest
from unittest.mock import patch

from src.health.checks import build_worker_health_payload


class WorkerHealthTestCase(unittest.TestCase):
    def test_worker_health_is_ok_when_all_checks_pass(self) -> None:
        with patch(
            "src.health.checks.check_worker_ping",
            return_value=("ok", {"status": "ok"}),
        ):
            with patch(
                "src.health.checks.check_queue",
                return_value=("ok", {"status": "ok"}),
            ):
                with patch(
                    "src.health.checks.check_storage",
                    return_value=("ok", {"status": "ok"}),
                ):
                    with patch(
                        "src.health.checks.check_model_config",
                        return_value=("ok", {"status": "ok"}),
                    ):
                        with patch(
                            "src.health.checks.check_model_preload",
                            return_value=("ok", {"status": "ok"}),
                        ):
                            status_code, payload = build_worker_health_payload()

        self.assertEqual(status_code, 0)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["service"], "inference-worker")
        self.assertEqual(payload["check_type"], "readiness")
        self.assertTrue(payload["ready"])
        self.assertEqual(payload["summary"]["total"], 5)
        self.assertEqual(payload["summary"]["passing"], 5)
        self.assertEqual(payload["summary"]["failing"], 0)
        self.assertEqual(payload["summary"]["failingChecks"], [])

    def test_worker_health_fails_when_ping_fails(self) -> None:
        with patch(
            "src.health.checks.check_worker_ping",
            return_value=("error", {"status": "error", "message": "no pong"}),
        ):
            with patch(
                "src.health.checks.check_queue",
                return_value=("ok", {"status": "ok"}),
            ):
                with patch(
                    "src.health.checks.check_storage",
                    return_value=("ok", {"status": "ok"}),
                ):
                    with patch(
                        "src.health.checks.check_model_config",
                        return_value=("ok", {"status": "ok"}),
                    ):
                        with patch(
                            "src.health.checks.check_model_preload",
                            return_value=("ok", {"status": "ok"}),
                        ):
                            status_code, payload = build_worker_health_payload()

        self.assertEqual(status_code, 1)
        self.assertEqual(payload["status"], "degraded")
        self.assertFalse(payload["ready"])
        self.assertEqual(payload["checks"]["worker"]["status"], "error")
        self.assertEqual(payload["summary"]["failingChecks"], ["worker"])
        self.assertEqual(payload["summary"]["statuses"]["worker"], "error")

    def test_worker_health_fails_when_preload_is_not_ready(self) -> None:
        with patch(
            "src.health.checks.check_worker_ping",
            return_value=("ok", {"status": "ok"}),
        ):
            with patch(
                "src.health.checks.check_queue",
                return_value=("ok", {"status": "ok"}),
            ):
                with patch(
                    "src.health.checks.check_storage",
                    return_value=("ok", {"status": "ok"}),
                ):
                    with patch(
                        "src.health.checks.check_model_config",
                        return_value=("ok", {"status": "ok"}),
                    ):
                        with patch(
                            "src.health.checks.check_model_preload",
                            return_value=(
                                "error",
                                {"status": "error", "message": "still loading"},
                            ),
                        ):
                            status_code, payload = build_worker_health_payload()

        self.assertEqual(status_code, 1)
        self.assertEqual(payload["status"], "degraded")
        self.assertFalse(payload["ready"])
        self.assertEqual(payload["checks"]["preload"]["status"], "error")
        self.assertEqual(payload["summary"]["failingChecks"], ["preload"])

    def test_worker_ping_uses_process_fallback(self) -> None:
        with patch("src.health.checks.Celery") as celery_cls:
            celery_cls.return_value.control.ping.return_value = []
            with patch(
                "src.health.checks._check_worker_process",
                return_value=("ok", {"status": "ok", "pid": 1}),
            ):
                from src.health.checks import check_worker_ping

                status, detail = check_worker_ping()

        self.assertEqual(status, "ok")
        self.assertEqual(detail["status"], "ok")
        self.assertIn("fallback_process", detail)


if __name__ == "__main__":
    unittest.main()
