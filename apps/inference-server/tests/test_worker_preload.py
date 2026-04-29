import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.worker_preload import preload_configured_model


class WorkerPreloadTestCase(unittest.TestCase):
    def test_preload_writes_ready_status_for_vlm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            status_path = Path(tmp_dir) / "preload.json"
            fake_model = type("FakeModel", (), {"_load_time_sec": 12.3456})()

            with patch.dict(
                "os.environ",
                {
                    "INFERENCE_PRELOAD_STATUS_PATH": str(status_path),
                    "VISION_CAPTION_MODEL": "qwen2.5-vl-7b",
                    "VISION_CAPTION_QUANTIZATION": "4bit",
                    "VISION_CAPTION_DTYPE": "float16",
                },
                clear=True,
            ):
                with patch(
                    "src.worker_preload.get_qwen_vlm_components",
                    return_value=("cuda", object(), fake_model, object()),
                ):
                    payload = preload_configured_model()

            self.assertEqual(payload["status"], "ready")
            self.assertEqual(payload["model_key"], "qwen2.5-vl-7b")
            self.assertEqual(payload["quantization"], "4bit")
            self.assertEqual(
                payload["executionPolicy"]["selectedModelKey"], "qwen2.5-vl-7b"
            )
            self.assertEqual(payload["executionPolicy"]["fallbackTriggered"], False)
            self.assertTrue(status_path.exists())

            saved = json.loads(status_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "ready")
            self.assertEqual(saved["model_key"], "qwen2.5-vl-7b")
            self.assertEqual(
                saved["executionPolicy"]["selectedModelKey"], "qwen2.5-vl-7b"
            )

    def test_preload_writes_error_status_when_loader_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            status_path = Path(tmp_dir) / "preload.json"

            with patch.dict(
                "os.environ",
                {
                    "INFERENCE_PRELOAD_STATUS_PATH": str(status_path),
                    "VISION_CAPTION_MODEL": "qwen2.5-vl-7b",
                },
                clear=True,
            ):
                with patch(
                    "src.worker_preload.get_qwen_vlm_components",
                    side_effect=RuntimeError("OOM"),
                ):
                    with self.assertRaises(RuntimeError):
                        preload_configured_model()

            saved = json.loads(status_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "error")
            self.assertEqual(saved["error"], "OOM")

    def test_preload_uses_serving_profile_when_runtime_env_is_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            status_path = Path(tmp_dir) / "preload.json"
            profile_path = Path(tmp_dir) / "serving-profile.json"
            fake_model = type("FakeModel", (), {"_load_time_sec": 1.2345})()

            profile_path.write_text(
                json.dumps(
                    {
                        "profile_type": "caption_serving_profile",
                        "generated_at_utc": "2026-04-08T00:00:00Z",
                        "default": {
                            "model_key": "blip-base",
                            "quantization": "none",
                            "dtype_name": "float16",
                        },
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(
                "os.environ",
                {
                    "INFERENCE_PRELOAD_STATUS_PATH": str(status_path),
                    "INFERENCE_SERVING_PROFILE_PATH": str(profile_path),
                },
                clear=True,
            ):
                with patch(
                    "src.worker_preload.get_caption_model_components",
                    return_value=("cpu", object(), fake_model, object()),
                ):
                    payload = preload_configured_model()

            self.assertEqual(payload["status"], "ready")
            self.assertEqual(payload["model_key"], "blip-base")
            self.assertEqual(payload["settings_source"], "profile")
            self.assertEqual(payload["profile_path"], str(profile_path))
            self.assertEqual(payload["executionPolicy"]["settingsSource"], "profile")
            self.assertEqual(payload["executionPolicy"]["profilePath"], str(profile_path))


if __name__ == "__main__":
    unittest.main()
