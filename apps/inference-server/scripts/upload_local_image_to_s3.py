import argparse
import mimetypes
from pathlib import Path

from src.storage.s3 import get_storage_service


def _guess_content_type(image_path: Path) -> str:
    guessed, _ = mimetypes.guess_type(str(image_path))
    return guessed or "image/jpeg"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upload a local image to object storage."
    )
    parser.add_argument("--image", required=True, help="Path to image inside container")
    parser.add_argument(
        "--image-key", required=True, help="Destination object storage key"
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with image_path.open("rb") as file_obj:
        get_storage_service().write_object(
            key=args.image_key,
            body=file_obj.read(),
            content_type=_guess_content_type(image_path),
        )
    print(args.image_key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
