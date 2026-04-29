import sys
import types
import unittest


def _install_test_stubs() -> None:
    torch_stub = types.ModuleType("torch")
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch_stub.dtype = type("dtype", (), {})
    torch_stub.float16 = object()
    torch_stub.bfloat16 = object()
    torch_stub.float32 = object()

    pil_stub = types.ModuleType("PIL")
    pil_stub.__path__ = []  # type: ignore[attr-defined]
    pil_image_stub = types.ModuleType("PIL.Image")
    pil_image_stub.Image = type("Image", (), {})

    class _StubGeneratedImage:
        def convert(self, mode):  # noqa: ANN001
            return self

        def save(self, buffer, format=None):  # noqa: ANN001
            buffer.write(b"stub-image-bytes")

    pil_image_stub.new = staticmethod(lambda *args, **kwargs: _StubGeneratedImage())
    pil_image_stub.open = staticmethod(lambda *args, **kwargs: _StubGeneratedImage())
    pil_stub.Image = pil_image_stub
    pil_stub.UnidentifiedImageError = type("UnidentifiedImageError", (Exception,), {})

    transformers_stub = types.ModuleType("transformers")
    for name in (
        "AutoProcessor",
        "AutoTokenizer",
        "BitsAndBytesConfig",
        "BlipForConditionalGeneration",
        "BlipProcessor",
        "GitForCausalLM",
        "VisionEncoderDecoderModel",
        "ViTImageProcessor",
    ):
        setattr(transformers_stub, name, type(name, (), {}))

    sys.modules.setdefault("torch", torch_stub)
    sys.modules.setdefault("PIL", pil_stub)
    sys.modules.setdefault("PIL.Image", pil_image_stub)
    sys.modules.setdefault("transformers", transformers_stub)


_install_test_stubs()

from src.contracts.vlm import build_vlm_error_result, build_vlm_success_result


class VlmContractErrorHandlingTestCase(unittest.TestCase):
    def test_build_vlm_success_result_normalizes_partial_metadata_payload(self) -> None:
        result = build_vlm_success_result(
            request_id="req-success-1",
            user_id="user-1",
            image_key="captures/sample.jpg",
            model_key="unknown-model",
            quantization="none",
            dtype_name="float16",
            generation_result={
                "caption": "wallet on the desk next to the keyboard",
                "elapsed_sec": 0.42,
                "peak_memory_mb": 128.0,
                "load_time_sec": 1.3,
            },
            captured_at="2026-04-08T09:00:00Z",
            inference_metadata={
                "caption": "  wallet on the desk next to the keyboard  ",
                "detectedObjects": ["wallet", "wallet", "keyboard", " "],
                "tags": ("desk", "keyboard", "desk"),
                "ocrText": "   ",
                "location": {
                    "name": " desk ",
                    "latitude": "37.5",
                    "longitude": "bad-number",
                },
            },
            execution_policy={
                "settingsSource": "env",
                "profilePath": None,
                "selectedModelKey": "unknown-model",
                "softTimeLimitSec": 120,
                "hardTimeLimitSec": 150,
                "fallbackModelKey": None,
                "fallbackTriggered": False,
            },
        )

        self.assertEqual(result["capturedAt"], "2026-04-08T09:00:00Z")
        self.assertEqual(
            result["providerMetadata"]["executionPolicy"]["selectedModelKey"],
            "unknown-model",
        )
        self.assertEqual(
            result["metadata"],
            {
                "caption": "wallet on the desk next to the keyboard",
                "sceneSummary": None,
                "detectedObjects": ["wallet", "keyboard"],
                "tags": ["desk", "keyboard"],
                "ocrText": None,
                "positionHint": "keyboard 옆",
                "location": {
                    "name": "desk",
                    "address": None,
                    "latitude": 37.5,
                    "longitude": None,
                },
            },
        )

    def test_build_vlm_error_result_handles_unknown_model_key(self) -> None:
        result = build_vlm_error_result(
            request_id="req-1",
            user_id="user-1",
            image_key="captures/sample.jpg",
            error=RuntimeError("boom"),
            model_key="unknown-model",
            quantization="none",
            dtype_name="float16",
            captured_at="2026-04-08T10:00:00Z",
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["capturedAt"], "2026-04-08T10:00:00Z")
        self.assertEqual(result["providerMetadata"]["modelKey"], "unknown-model")
        self.assertEqual(result["providerMetadata"]["modelId"], "unknown-model")
        self.assertIsNone(result["providerMetadata"]["modelFamily"])

    def test_build_vlm_error_result_accepts_explicit_error_policy(self) -> None:
        result = build_vlm_error_result(
            request_id="req-2",
            user_id="user-2",
            image_key="captures/sample.jpg",
            error=RuntimeError("timeout"),
            model_key="unknown-model",
            quantization="none",
            dtype_name="float16",
            error_code="inference_timeout",
            retryable=True,
        )

        self.assertEqual(result["errorCode"], "inference_timeout")
        self.assertTrue(result["retryable"])

    def test_build_vlm_error_result_accepts_error_details(self) -> None:
        result = build_vlm_error_result(
            request_id="req-3",
            user_id="user-3",
            image_key="captures/sample.jpg",
            error=TimeoutError("timed out"),
            model_key="unknown-model",
            quantization="none",
            dtype_name="float16",
            error_code="inference_timeout",
            retryable=True,
            error_details={
                "category": " timeout ",
                "reason": "inference_timeout",
                "exceptionType": "TimeoutError",
                "retryable": True,
                "source": " inference_worker ",
                "taskTimeLimit": {
                    "softSec": "120",
                    "hardSec": 150,
                },
            },
        )

        self.assertEqual(
            result["errorDetails"],
            {
                "category": "timeout",
                "reason": "inference_timeout",
                "exceptionType": "TimeoutError",
                "retryable": True,
                "source": "inference_worker",
                "taskTimeLimit": {"softSec": 120, "hardSec": 150},
            },
        )


if __name__ == "__main__":
    unittest.main()
