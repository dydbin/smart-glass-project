import argparse
import json
import mimetypes
import os
from pathlib import Path

from src.queue.tasks import process_vision_inference
from src.models.qwen_vlm import get_qwen_vlm_spec
from src.storage.s3 import get_storage_service


def _guess_content_type(image_path: Path) -> str:
    guessed, _ = mimetypes.guess_type(str(image_path))
    return guessed or "image/jpeg"


def _upload_sample_image(image_path: Path, image_key: str) -> None:
    with image_path.open("rb") as file_obj:
        get_storage_service().write_object(
            key=image_key,
            body=file_obj.read(),
            content_type=_guess_content_type(image_path),
        )


def _validate_result(result: dict) -> None:
    required_top_level_keys = {
        "status",
        "requestId",
        "taskType",
        "userId",
        "sourceImage",
        "metadata",
        "providerMetadata",
        "runtime",
    }
    missing = sorted(required_top_level_keys - set(result))
    if missing:
        raise AssertionError(f"Missing top-level keys: {missing}")
    if result["status"] != "success":
        raise AssertionError(f"Expected success result, got: {json.dumps(result)}")

    metadata = result["metadata"]
    for key in (
        "caption",
        "sceneSummary",
        "detectedObjects",
        "tags",
        "ocrText",
        "positionHint",
        "location",
    ):
        if key not in metadata:
            raise AssertionError(f"Missing metadata key: {key}")

    provider_metadata = result["providerMetadata"]
    expected_model_key = os.getenv("VISION_CAPTION_MODEL", "qwen2.5-vl-3b")
    if provider_metadata.get("modelKey") != expected_model_key:
        raise AssertionError(
            f"Unexpected modelKey: {provider_metadata.get('modelKey')}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Qwen VLM smoke test and validate VlmInferenceResult."
    )
    parser.add_argument(
        "--image",
        default="/app/sample_data/key_1.jpg",
        help="Path to sample image inside container",
    )
    parser.add_argument(
        "--image-key",
        default="smoke-tests/qwen/key_1.jpg",
        help="Object storage key to upload before inference",
    )
    parser.add_argument("--user-id", default="smoke-user")
    parser.add_argument("--memory-id", default="smoke-memory")
    parser.add_argument("--request-id", default="smoke-qwen-req-1")
    parser.add_argument("--model-key", default="qwen2.5-vl-3b")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Sample image not found: {image_path}")

    get_qwen_vlm_spec(args.model_key)
    os.environ["VISION_CAPTION_MODEL"] = args.model_key
    os.environ.setdefault("VISION_CAPTION_QUANTIZATION", "4bit")
    os.environ.setdefault("VISION_CAPTION_DTYPE", "float16")

    _upload_sample_image(image_path, args.image_key)
    result = process_vision_inference(
        image_key=args.image_key,
        user_id=args.user_id,
        memory_id=args.memory_id,
        request_id=args.request_id,
        task_type="metadata",
    )
    _validate_result(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
