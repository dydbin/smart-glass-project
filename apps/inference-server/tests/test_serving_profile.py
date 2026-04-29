import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.models.serving_profile import (
    build_caption_serving_profile,
    resolve_runtime_execution_policy,
    resolve_runtime_serving_settings,
)


class ServingProfileTestCase(unittest.TestCase):
    def _write_profile(self, path: Path, *, model_key: str, quantization: str, dtype_name: str) -> None:
        path.write_text(
            json.dumps(
                {
                    "profile_type": "caption_serving_profile",
                    "generated_at_utc": "2026-04-08T00:00:00Z",
                    "default": {
                        "model_key": model_key,
                        "quantization": quantization,
                        "dtype_name": dtype_name,
                    },
                }
            ),
            encoding="utf-8",
        )

    def test_build_caption_serving_profile_selects_default_and_fallback(self) -> None:
        summary_payload = {
            "generated_at_utc": "2026-04-08T00:00:00Z",
            "results": [
                {
                    "model_key": "blip-base",
                    "model_id": "Salesforce/blip-image-captioning-base",
                    "quantization": "none",
                    "dtype": "float16",
                    "images_total": 10,
                    "images_success": 10,
                    "images_error": 0,
                    "load_time_sec": 12.1,
                    "latency_p95_sec": 0.5,
                    "peak_memory_max_mb": 520.0,
                    "lexical_f1_avg": 0.52,
                },
                {
                    "model_key": "git-base",
                    "model_id": "microsoft/git-base-coco",
                    "quantization": "none",
                    "dtype": "float16",
                    "images_total": 10,
                    "images_success": 10,
                    "images_error": 0,
                    "load_time_sec": 20.0,
                    "latency_p95_sec": 0.7,
                    "peak_memory_max_mb": 820.0,
                    "lexical_f1_avg": 0.61,
                },
                {
                    "model_key": "vit-gpt2",
                    "model_id": "nlpconnect/vit-gpt2-image-captioning",
                    "quantization": "none",
                    "dtype": "float16",
                    "images_total": 10,
                    "images_success": 8,
                    "images_error": 2,
                    "load_time_sec": 22.0,
                    "latency_p95_sec": 0.3,
                    "peak_memory_max_mb": 1300.0,
                    "lexical_f1_avg": 0.49,
                },
            ],
        }

        profile = build_caption_serving_profile(summary_payload, latency_budget_sec=10.0)

        self.assertEqual(profile["default"]["model_key"], "git-base")
        self.assertEqual(profile["fallback"]["model_key"], "blip-base")
        self.assertEqual(len(profile["candidates"]), 2)
        self.assertTrue(profile["default"]["meets_latency_budget"])

    def test_resolve_runtime_serving_settings_prefers_explicit_env_over_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            profile_path = Path(tmp_dir) / "serving-profile.json"
            self._write_profile(
                profile_path,
                model_key="blip-base",
                quantization="none",
                dtype_name="bfloat16",
            )

            with patch.dict(
                "os.environ",
                {
                    "INFERENCE_SERVING_PROFILE_PATH": str(profile_path),
                    "VISION_CAPTION_MODEL": "qwen2.5-vl-7b",
                    "VISION_CAPTION_QUANTIZATION": "4bit",
                },
                clear=True,
            ):
                resolved = resolve_runtime_serving_settings()

        self.assertEqual(resolved.source, "env")
        self.assertEqual(resolved.model_key, "qwen2.5-vl-7b")
        self.assertEqual(resolved.quantization, "4bit")
        self.assertEqual(resolved.dtype_name, "float16")
        self.assertIsNone(resolved.profile_path)

    def test_resolve_runtime_serving_settings_uses_profile_when_env_is_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            profile_path = Path(tmp_dir) / "serving-profile.json"
            self._write_profile(
                profile_path,
                model_key="git-base",
                quantization="none",
                dtype_name="float16",
            )

            with patch.dict(
                "os.environ",
                {"INFERENCE_SERVING_PROFILE_PATH": str(profile_path)},
                clear=True,
            ):
                resolved = resolve_runtime_serving_settings()

        self.assertEqual(resolved.source, "profile")
        self.assertEqual(resolved.model_key, "git-base")
        self.assertEqual(resolved.profile_path, str(profile_path))

    def test_resolve_runtime_serving_settings_raises_for_missing_configured_profile(self) -> None:
        with patch.dict(
            "os.environ",
            {"INFERENCE_SERVING_PROFILE_PATH": "/tmp/does-not-exist.json"},
            clear=True,
        ):
            with self.assertRaises(FileNotFoundError):
                resolve_runtime_serving_settings()

    def test_resolve_runtime_execution_policy_includes_time_limits_and_vlm_fallback(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "VISION_CAPTION_MODEL": "qwen2.5-vl-7b",
                "VISION_CAPTION_QUANTIZATION": "4bit",
                "VISION_TASK_SOFT_TIME_LIMIT_SEC": "90",
                "VISION_TASK_HARD_TIME_LIMIT_SEC": "120",
                "VISION_QWEN_FALLBACK_MODEL": "qwen2.5-vl-3b",
            },
            clear=True,
        ):
            policy = resolve_runtime_execution_policy()

        self.assertEqual(policy.settings.source, "env")
        self.assertEqual(policy.settings.model_key, "qwen2.5-vl-7b")
        self.assertEqual(policy.soft_time_limit_sec, 90)
        self.assertEqual(policy.hard_time_limit_sec, 120)
        self.assertEqual(policy.fallback_model_key, "qwen2.5-vl-3b")
        self.assertEqual(
            policy.to_payload(),
            {
                "settingsSource": "env",
                "profilePath": None,
                "selectedModelKey": "qwen2.5-vl-7b",
                "softTimeLimitSec": 90,
                "hardTimeLimitSec": 120,
                "fallbackModelKey": "qwen2.5-vl-3b",
                "fallbackTriggered": False,
            },
        )


if __name__ == "__main__":
    unittest.main()
