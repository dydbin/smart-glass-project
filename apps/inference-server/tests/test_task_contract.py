import io
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from celery.exceptions import SoftTimeLimitExceeded
from PIL import Image

from src.queue.tasks import _should_retry_task, process_vision_inference
from src.storage.s3 import StorageAccessError, StorageNotFoundError, StorageObject


class _FakeStorageService:
    def __init__(self, payload: bytes, content_type: str = "image/jpeg"):
        self.payload = payload
        self.content_type = content_type

    def read_object(self, key: str, *, bucket_name: str | None = None) -> StorageObject:
        return StorageObject(
            bucket_name=(
                bucket_name
                or os.getenv("STORAGE_BUCKET_NAME")
                or os.getenv("AWS_S3_BUCKET_NAME", "smart-glass-test")
            ),
            key=key,
            body=self.payload,
            content_type=self.content_type,
        )


class _FailingStorageService:
    def __init__(self, error: Exception):
        self.error = error

    def read_object(self, key: str, *, bucket_name: str | None = None) -> StorageObject:
        raise self.error


def _build_test_image_bytes() -> bytes:
    image = Image.new("RGB", (4, 4), color=(255, 255, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


class TaskContractTestCase(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["STORAGE_BUCKET_NAME"] = "smart-glass-test"
        os.environ["VISION_CAPTION_MODEL"] = "blip-base"
        os.environ["VISION_CAPTION_QUANTIZATION"] = "none"
        os.environ["VISION_CAPTION_DTYPE"] = "float16"
        os.environ.pop("VISION_PROVIDER_METADATA_INCLUDE_RAW", None)

    def tearDown(self) -> None:
        for name in (
            "STORAGE_BUCKET_NAME",
            "AWS_S3_BUCKET_NAME",
            "VISION_CAPTION_MODEL",
            "VISION_CAPTION_QUANTIZATION",
            "VISION_CAPTION_DTYPE",
            "VISION_PROVIDER_METADATA_INCLUDE_RAW",
            "VISION_QWEN_FALLBACK_MODEL",
        ):
            os.environ.pop(name, None)

    def test_process_vision_inference_returns_vlm_success_contract(self) -> None:
        with patch(
            "src.queue.tasks.get_storage_service",
            return_value=_FakeStorageService(_build_test_image_bytes()),
        ):
            with patch(
                "src.queue.tasks.generate_caption",
                return_value={
                    "caption": "a wallet is on the desk next to the keyboard",
                    "elapsed_sec": 0.42,
                    "peak_memory_mb": 512.5,
                    "load_time_sec": 1.2,
                    "model_id": "Salesforce/blip-image-captioning-base",
                    "model_key": "blip-base",
                    "device": "cuda",
                    "quantization": "none",
                    "prompt": "a photography of",
                },
            ):
                result = process_vision_inference(
                    image_key="captures/wallet-01.jpg",
                    user_id="user-1",
                    memory_id="mem-1",
                    captured_at="2026-04-04T10:00:00Z",
                    image_url="https://example.com/captures/wallet-01.jpg",
                    request_id="req-123",
                )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["requestId"], "req-123")
        self.assertEqual(result["memoryId"], "mem-1")
        self.assertEqual(result["userId"], "user-1")
        self.assertEqual(result["sourceImage"]["imageKey"], "captures/wallet-01.jpg")
        self.assertEqual(result["sourceImage"]["contentType"], "image/jpeg")
        self.assertEqual(
            result["metadata"]["caption"],
            "a wallet is on the desk next to the keyboard",
        )
        self.assertEqual(result["metadata"]["detectedObjects"], [])
        self.assertEqual(result["metadata"]["tags"], [])
        self.assertEqual(result["metadata"]["positionHint"], "keyboard 옆")
        self.assertEqual(result["providerMetadata"]["modelKey"], "blip-base")
        self.assertEqual(result["providerMetadata"]["capabilities"]["caption"], True)
        self.assertEqual(
            result["providerMetadata"]["executionPolicy"]["settingsSource"], "env"
        )
        self.assertEqual(
            result["providerMetadata"]["executionPolicy"]["selectedModelKey"],
            "blip-base",
        )
        self.assertEqual(
            result["providerMetadata"]["executionPolicy"]["fallbackTriggered"], False
        )
        self.assertEqual(
            result["providerMetadata"]["capabilities"]["sceneSummary"], False
        )
        self.assertEqual(
            result["providerMetadata"]["capabilities"]["detectedObjects"], False
        )
        self.assertEqual(
            result["providerMetadata"]["capabilities"]["pipelineOutput"], False
        )
        self.assertEqual(result["providerMetadata"]["raw"], None)
        self.assertEqual(result["runtime"]["latencySec"], 0.42)
        self.assertEqual(result["runtime"]["peakMemoryMb"], 512.5)

    def test_process_vision_inference_returns_vlm_error_contract(self) -> None:
        with patch(
            "src.queue.tasks.get_storage_service",
            return_value=_FailingStorageService(StorageAccessError("s3 unavailable")),
        ):
            result = process_vision_inference(
                image_key="captures/wallet-01.jpg",
                user_id="user-1",
                memory_id="mem-1",
                captured_at="2026-04-04T10:00:00Z",
                request_id="req-err-1",
            )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["requestId"], "req-err-1")
        self.assertEqual(result["capturedAt"], "2026-04-04T10:00:00Z")
        self.assertEqual(result["errorCode"], "storage_access_error")
        self.assertEqual(result["message"], "s3 unavailable")
        self.assertTrue(result["retryable"])
        self.assertEqual(result["errorDetails"]["category"], "storage")
        self.assertEqual(result["errorDetails"]["reason"], "storage_access_error")
        self.assertEqual(result["errorDetails"]["exceptionType"], "StorageAccessError")
        self.assertTrue(result["errorDetails"]["retryable"])
        self.assertEqual(result["errorDetails"]["source"], "inference_worker")
        self.assertEqual(result["providerMetadata"]["modelKey"], "blip-base")
        self.assertEqual(
            result["providerMetadata"]["executionPolicy"]["selectedModelKey"],
            "blip-base",
        )
        self.assertEqual(
            result["providerMetadata"]["executionPolicy"]["softTimeLimitSec"], 120
        )
        self.assertEqual(
            result["providerMetadata"]["capabilities"]["sceneSummary"], False
        )

    def test_process_vision_inference_marks_missing_source_image_as_non_retryable(
        self,
    ) -> None:
        with patch(
            "src.queue.tasks.get_storage_service",
            return_value=_FailingStorageService(
                StorageNotFoundError("captures/missing.jpg not found")
            ),
        ):
            result = process_vision_inference(
                image_key="captures/missing.jpg",
                user_id="user-1",
                request_id="req-missing-1",
            )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["errorCode"], "source_image_not_found")
        self.assertFalse(result["retryable"])
        self.assertEqual(result["errorDetails"]["category"], "source_image")
        self.assertEqual(result["errorDetails"]["reason"], "source_image_not_found")
        self.assertFalse(result["errorDetails"]["retryable"])

    def test_process_vision_inference_marks_invalid_image_as_input_error(self) -> None:
        with patch(
            "src.queue.tasks.get_storage_service",
            return_value=_FakeStorageService(b"not-an-image"),
        ):
            result = process_vision_inference(
                image_key="captures/broken.jpg",
                user_id="user-1",
                request_id="req-invalid-image-1",
            )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["errorCode"], "invalid_source_image")
        self.assertFalse(result["retryable"])
        self.assertEqual(result["errorDetails"]["category"], "input")
        self.assertEqual(result["errorDetails"]["reason"], "invalid_source_image")
        self.assertFalse(result["errorDetails"]["retryable"])

    def test_process_vision_inference_marks_invalid_object_key_as_non_retryable(
        self,
    ) -> None:
        with patch(
            "src.queue.tasks.get_storage_service",
            return_value=_FailingStorageService(
                ValueError("storage object key must not be blank")
            ),
        ):
            result = process_vision_inference(
                image_key="   ",
                user_id="user-1",
                request_id="req-invalid-key-1",
            )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["errorCode"], "invalid_inference_request")
        self.assertFalse(result["retryable"])

    def test_process_vision_inference_marks_timeout_as_retryable_timeout_error(self) -> None:
        with patch(
            "src.queue.tasks.get_storage_service",
            return_value=_FakeStorageService(_build_test_image_bytes()),
        ):
            with patch(
                "src.queue.tasks.generate_caption",
                side_effect=SoftTimeLimitExceeded(),
            ):
                result = process_vision_inference(
                    image_key="captures/wallet-01.jpg",
                    user_id="user-1",
                    request_id="req-timeout-1",
                )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["errorCode"], "inference_timeout")
        self.assertTrue(result["retryable"])
        self.assertEqual(result["errorDetails"]["category"], "timeout")
        self.assertEqual(result["errorDetails"]["reason"], "inference_timeout")
        self.assertEqual(
            result["errorDetails"]["exceptionType"],
            "SoftTimeLimitExceeded",
        )
        self.assertTrue(result["errorDetails"]["retryable"])
        self.assertEqual(
            result["errorDetails"]["taskTimeLimit"],
            {"softSec": 120, "hardSec": 150},
        )

    def test_worker_retry_policy_skips_direct_calls(self) -> None:
        task_request = SimpleNamespace(called_directly=True, retries=0)

        should_retry = _should_retry_task(
            retryable=True,
            task_request=task_request,
            max_retries=2,
        )

        self.assertFalse(should_retry)

    def test_worker_retry_policy_allows_retryable_worker_attempt(self) -> None:
        task_request = SimpleNamespace(called_directly=False, retries=1)

        should_retry = _should_retry_task(
            retryable=True,
            task_request=task_request,
            max_retries=2,
        )

        self.assertTrue(should_retry)

    def test_worker_retry_policy_stops_at_max_retries(self) -> None:
        task_request = SimpleNamespace(called_directly=False, retries=2)

        should_retry = _should_retry_task(
            retryable=True,
            task_request=task_request,
            max_retries=2,
        )

        self.assertFalse(should_retry)

    def test_worker_retry_policy_skips_non_retryable_errors(self) -> None:
        task_request = SimpleNamespace(called_directly=False, retries=0)

        should_retry = _should_retry_task(
            retryable=False,
            task_request=task_request,
            max_retries=2,
        )

        self.assertFalse(should_retry)

    def test_process_vision_inference_routes_qwen_vlm_metadata_to_contract(self) -> None:
        os.environ["VISION_CAPTION_MODEL"] = "qwen2.5-vl-7b"
        os.environ["VISION_CAPTION_QUANTIZATION"] = "4bit"

        with patch(
            "src.queue.tasks.get_storage_service",
            return_value=_FakeStorageService(_build_test_image_bytes()),
        ):
            with patch(
                "src.queue.tasks.generate_qwen_vlm_metadata",
                return_value={
                    "metadata": {
                        "caption": "지갑이 키보드 옆 책상 위에 놓여 있다",
                        "sceneSummary": "작업용 책상 장면",
                        "detectedObjects": ["지갑", "키보드"],
                        "tags": ["지갑", "키보드", "책상"],
                        "ocrText": None,
                        "positionHint": "키보드 옆",
                        "location": None,
                    },
                    "pipeline_output": {
                        "capture_id": "capture-1",
                        "timestamp": "2026-04-06T14:00:00",
                        "image_path": None,
                        "sharpness_score": 123.45,
                        "inference_time": 1.17,
                        "scene_summary": "작업용 책상 장면",
                        "location_context": "사무실",
                        "objects": [
                            {
                                "object_id": 1,
                                "name": "지갑",
                                "confidence": 0.8,
                                "position": {
                                    "depth_hint": "near",
                                    "surface": "책상 위",
                                },
                                "visual_features": {
                                    "color": "검정",
                                    "material": "가죽",
                                    "brand": None,
                                    "shape": "직사각형",
                                },
                                "nearby_objects": ["키보드"],
                                "raw_description_ko": "키보드 옆 지갑",
                            }
                        ],
                        "pipeline_meta": {
                            "vlm_model": "Qwen/Qwen2.5-VL-7B-Instruct",
                            "object_count": 1,
                            "vram_allocated_gb": 5.2,
                            "vram_peak_gb": 5.4,
                            "vram_total_gb": 8.0,
                        },
                    },
                    "elapsed_sec": 1.17,
                    "peak_memory_mb": 2048.0,
                    "load_time_sec": 8.5,
                    "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
                    "model_key": "qwen2.5-vl-7b",
                    "device": "cuda",
                    "quantization": "4bit",
                    "prompt": "structured prompt",
                    "system_prompt": "system",
                    "raw_output_text": "{\"caption\":\"지갑이 키보드 옆 책상 위에 놓여 있다\"}",
                },
            ):
                result = process_vision_inference(
                    image_key="captures/wallet-02.jpg",
                    user_id="user-2",
                    memory_id="mem-2",
                    request_id="req-qwen-1",
                )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["requestId"], "req-qwen-1")
        self.assertEqual(result["metadata"]["caption"], "지갑이 키보드 옆 책상 위에 놓여 있다")
        self.assertEqual(result["metadata"]["sceneSummary"], "작업용 책상 장면")
        self.assertEqual(result["metadata"]["detectedObjects"], ["지갑", "키보드"])
        self.assertEqual(result["metadata"]["positionHint"], "키보드 옆")
        self.assertEqual(result["pipelineOutput"]["capture_id"], "capture-1")
        self.assertEqual(result["pipelineOutput"]["objects"][0]["name"], "지갑")
        self.assertEqual(result["providerMetadata"]["modelKey"], "qwen2.5-vl-7b")
        self.assertEqual(result["providerMetadata"]["modelFamily"], "qwen2_5_vl")
        self.assertEqual(
            result["providerMetadata"]["executionPolicy"]["selectedModelKey"],
            "qwen2.5-vl-7b",
        )
        self.assertEqual(
            result["providerMetadata"]["executionPolicy"]["fallbackTriggered"], False
        )
        self.assertEqual(
            result["providerMetadata"]["executionPolicy"]["fallbackModelKey"],
            "qwen2.5-vl-3b",
        )
        self.assertEqual(result["providerMetadata"]["capabilities"]["sceneSummary"], True)
        self.assertEqual(
            result["providerMetadata"]["capabilities"]["detectedObjects"], True
        )
        self.assertEqual(result["providerMetadata"]["capabilities"]["tags"], True)
        self.assertEqual(result["providerMetadata"]["capabilities"]["ocrText"], False)
        self.assertEqual(
            result["providerMetadata"]["capabilities"]["pipelineOutput"], True
        )
        self.assertEqual(result["runtime"]["peakMemoryMb"], 2048.0)

    def test_process_vision_inference_retries_with_qwen_fallback_model(self) -> None:
        os.environ["VISION_CAPTION_MODEL"] = "qwen2.5-vl-7b"
        os.environ["VISION_CAPTION_QUANTIZATION"] = "4bit"
        os.environ["VISION_QWEN_FALLBACK_MODEL"] = "qwen2.5-vl-3b"

        with patch(
            "src.queue.tasks.get_storage_service",
            return_value=_FakeStorageService(_build_test_image_bytes()),
        ):
            with patch(
                "src.queue.tasks.generate_qwen_vlm_metadata",
                side_effect=[
                    RuntimeError("CUDA out of memory"),
                    {
                        "metadata": {
                            "caption": "맥북과 아이폰이 책상 위에 있다",
                            "sceneSummary": "전자기기 책상 장면",
                            "detectedObjects": ["맥북", "아이폰"],
                            "tags": ["맥북", "아이폰", "책상 위"],
                            "ocrText": None,
                            "positionHint": "책상 위",
                            "location": None,
                        },
                        "pipeline_output": {
                            "capture_id": "capture-fallback",
                            "timestamp": "2026-04-06T14:00:00",
                            "image_path": None,
                            "sharpness_score": 111.0,
                            "inference_time": 2.34,
                            "scene_summary": "전자기기 책상 장면",
                            "location_context": "작업 공간",
                            "objects": [],
                            "pipeline_meta": {
                                "vlm_model": "Qwen/Qwen2.5-VL-3B-Instruct",
                                "object_count": 0,
                                "vram_allocated_gb": 3.0,
                                "vram_peak_gb": 3.2,
                                "vram_total_gb": 8.0,
                            },
                        },
                        "elapsed_sec": 2.34,
                        "peak_memory_mb": 1536.0,
                        "load_time_sec": 12.0,
                        "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
                        "model_key": "qwen2.5-vl-3b",
                        "device": "cuda",
                        "quantization": "4bit",
                        "prompt": "structured prompt",
                        "system_prompt": "system",
                        "raw_output_text": "{\"caption\":\"맥북과 아이폰이 책상 위에 있다\"}",
                    },
                ],
            ) as mocked_generate:
                result = process_vision_inference(
                    image_key="captures/desk-01.jpg",
                    user_id="user-3",
                    request_id="req-qwen-fallback-1",
                )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["providerMetadata"]["modelKey"], "qwen2.5-vl-3b")
        self.assertEqual(
            result["providerMetadata"]["executionPolicy"]["selectedModelKey"],
            "qwen2.5-vl-7b",
        )
        self.assertEqual(
            result["providerMetadata"]["executionPolicy"]["fallbackModelKey"],
            "qwen2.5-vl-3b",
        )
        self.assertEqual(
            result["providerMetadata"]["executionPolicy"]["fallbackTriggered"], True
        )
        self.assertEqual(result["providerMetadata"]["capabilities"]["sceneSummary"], True)
        self.assertEqual(result["metadata"]["detectedObjects"], ["맥북", "아이폰"])
        self.assertEqual(result["pipelineOutput"]["capture_id"], "capture-fallback")
        self.assertEqual(mocked_generate.call_count, 2)
        self.assertEqual(mocked_generate.call_args_list[0].kwargs["model_key"], "qwen2.5-vl-7b")
        self.assertEqual(mocked_generate.call_args_list[1].kwargs["model_key"], "qwen2.5-vl-3b")
