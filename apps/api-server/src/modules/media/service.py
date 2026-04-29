from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

from src.database.memory_store import MemoryRecord, PostgresMemoryStoreClient

try:  # pragma: no cover - optional runtime dependency
    import boto3
    from botocore.config import Config
    from botocore.exceptions import BotoCoreError, ClientError
except ImportError:  # pragma: no cover - handled at runtime
    boto3 = None
    Config = None
    BotoCoreError = Exception
    ClientError = Exception


class MediaUrlSignerError(RuntimeError):
    pass


class MediaUrlSignerConfigError(MediaUrlSignerError):
    pass


class MediaUrlSignerUnavailableError(MediaUrlSignerError):
    pass


@dataclass(frozen=True, slots=True)
class MediaAccessUrl:
    image_key: str
    access_url: str
    expires_at: str
    expires_in_sec: int


@dataclass(frozen=True, slots=True)
class GalleryItem:
    memory_id: str
    image_key: str
    image_url: str | None
    captured_at: str | None
    caption: str | None
    scene_summary: str | None
    position_hint: str | None


class MediaAccessRepository(Protocol):
    def list_by_user(
        self,
        user_id: str,
        *,
        limit: int | None = None,
    ) -> list[MemoryRecord]: ...

    def get_by_image_key(self, user_id: str, image_key: str) -> MemoryRecord | None: ...

    def list_by_image_keys(
        self,
        user_id: str,
        image_keys: list[str],
    ) -> list[MemoryRecord]: ...

    def check_health(self) -> None: ...


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def _require_any_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    raise MediaUrlSignerConfigError(
        f"Missing required environment variable: one of {', '.join(names)}"
    )


def _resolve_bucket_name(bucket_name: str | None = None) -> str:
    normalized = (
        bucket_name
        or os.getenv("STORAGE_BUCKET_NAME")
        or os.getenv("AWS_S3_BUCKET_NAME")
        or ""
    ).strip()
    if not normalized:
        raise MediaUrlSignerConfigError(
            "Missing required environment variable: STORAGE_BUCKET_NAME"
        )
    return normalized


def _resolve_region_name() -> str:
    return (os.getenv("STORAGE_REGION") or os.getenv("AWS_REGION") or "ap-northeast-2").strip() or "ap-northeast-2"


def _resolve_endpoint_url() -> str | None:
    endpoint_url = os.getenv("STORAGE_ENDPOINT_URL", "").strip()
    return endpoint_url or None


def _resolve_addressing_style() -> str:
    return os.getenv("STORAGE_ADDRESSING_STYLE", "auto").strip().lower() or "auto"


def _build_s3_client():
    if boto3 is None:
        raise MediaUrlSignerUnavailableError(
            "boto3 is required for media access URL signing. Install apps/api-server requirements."
        )

    client_kwargs: dict[str, Any] = {
        "service_name": "s3",
        "aws_access_key_id": _require_any_env(
            "STORAGE_ACCESS_KEY_ID",
            "AWS_ACCESS_KEY_ID",
        ),
        "aws_secret_access_key": _require_any_env(
            "STORAGE_SECRET_ACCESS_KEY",
            "AWS_SECRET_ACCESS_KEY",
        ),
        "region_name": _resolve_region_name(),
    }
    endpoint_url = _resolve_endpoint_url()
    if endpoint_url:
        client_kwargs["endpoint_url"] = endpoint_url

    addressing_style = _resolve_addressing_style()
    if addressing_style != "auto":
        if Config is None:
            raise MediaUrlSignerUnavailableError(
                "botocore Config is unavailable for storage URL signing."
            )
        client_kwargs["config"] = Config(s3={"addressing_style": addressing_style})

    return boto3.client(**client_kwargs)


class S3MediaUrlSigner:
    def __init__(
        self,
        *,
        client: Any | None = None,
        default_bucket_name: str | None = None,
        default_expiration_sec: int = 300,
    ) -> None:
        self._client = client
        self._default_bucket_name = default_bucket_name
        self._default_expiration_sec = self._resolve_expiration(default_expiration_sec)

    def _get_client(self):
        if self._client is None:
            self._client = _build_s3_client()
        return self._client

    def _resolve_bucket_name(self, bucket_name: str | None = None) -> str:
        return _resolve_bucket_name(bucket_name or self._default_bucket_name)

    def _resolve_expiration(self, expires_in_sec: int | None) -> int:
        if expires_in_sec is None:
            expires_in_sec = self._default_expiration_sec if hasattr(self, "_default_expiration_sec") else 300
        try:
            resolved = int(expires_in_sec)
        except (TypeError, ValueError) as exc:
            raise ValueError("expiresInSec must be an integer between 30 and 3600") from exc
        if resolved < 30 or resolved > 3600:
            raise ValueError("expiresInSec must be between 30 and 3600")
        return resolved

    def sign_get_object(
        self,
        image_key: str,
        *,
        expires_in_sec: int | None = None,
        bucket_name: str | None = None,
    ) -> MediaAccessUrl:
        normalized_key = _normalize_text(image_key)
        if not normalized_key:
            raise ValueError("imageKey must not be blank")

        resolved_expiration = self._resolve_expiration(expires_in_sec)
        resolved_bucket_name = self._resolve_bucket_name(bucket_name)

        try:
            access_url = self._get_client().generate_presigned_url(
                "get_object",
                Params={"Bucket": resolved_bucket_name, "Key": normalized_key},
                ExpiresIn=resolved_expiration,
            )
        except (ClientError, BotoCoreError, MediaUrlSignerError) as exc:
            raise MediaUrlSignerUnavailableError(
                f"Failed to generate media access URL: {exc}"
            ) from exc
        except Exception as exc:
            raise MediaUrlSignerUnavailableError(
                f"Unexpected error while generating media access URL: {exc}"
            ) from exc

        expires_at = (
            datetime.now(timezone.utc)
            .replace(microsecond=0)
            + timedelta(seconds=resolved_expiration)
        ).isoformat().replace("+00:00", "Z")
        return MediaAccessUrl(
            image_key=normalized_key,
            access_url=access_url,
            expires_at=expires_at,
            expires_in_sec=resolved_expiration,
        )

    def check_health(self) -> None:
        self._resolve_bucket_name()
        self._get_client()


class MediaAccessService:
    def __init__(
        self,
        repository: MediaAccessRepository,
        signer: S3MediaUrlSigner,
        *,
        default_gallery_limit: int = 50,
    ) -> None:
        self.repository = repository
        self.signer = signer
        self.default_gallery_limit = max(1, min(default_gallery_limit, 200))

    def _normalize_user_id(self, user_id: str) -> str:
        normalized_user_id = _normalize_text(user_id)
        if not normalized_user_id:
            raise ValueError("userId must not be blank")
        return normalized_user_id

    def issue_access_url(
        self,
        *,
        user_id: str,
        image_key: str,
        expires_in_sec: int | None = None,
    ) -> MediaAccessUrl:
        normalized_user_id = self._normalize_user_id(user_id)
        normalized_image_key = _normalize_text(image_key)
        if not normalized_image_key:
            raise ValueError("imageKey must not be blank")

        record = self.repository.get_by_image_key(normalized_user_id, normalized_image_key)
        if record is None:
            raise PermissionError("imageKey does not belong to the requested user")
        return self.signer.sign_get_object(
            record.image_key or normalized_image_key,
            expires_in_sec=expires_in_sec,
        )

    def issue_access_urls(
        self,
        *,
        user_id: str,
        image_keys: list[str],
        expires_in_sec: int | None = None,
    ) -> list[MediaAccessUrl]:
        normalized_user_id = self._normalize_user_id(user_id)
        if not image_keys:
            return []

        ordered_keys: list[str] = []
        seen: set[str] = set()
        for image_key in image_keys:
            normalized_image_key = _normalize_text(image_key)
            if not normalized_image_key or normalized_image_key in seen:
                continue
            ordered_keys.append(normalized_image_key)
            seen.add(normalized_image_key)
        if not ordered_keys:
            return []

        records = self.repository.list_by_image_keys(normalized_user_id, ordered_keys)
        record_by_image_key = {
            record.image_key: record
            for record in records
            if record.image_key
        }

        missing_keys = [
            image_key for image_key in ordered_keys if image_key not in record_by_image_key
        ]
        if missing_keys:
            raise PermissionError("One or more imageKeys do not belong to the requested user")

        return [
            self.signer.sign_get_object(
                record_by_image_key[image_key].image_key or image_key,
                expires_in_sec=expires_in_sec,
            )
            for image_key in ordered_keys
        ]

    def list_gallery_items(
        self,
        *,
        user_id: str,
        limit: int | None = None,
    ) -> list[GalleryItem]:
        normalized_user_id = self._normalize_user_id(user_id)
        resolved_limit = limit if isinstance(limit, int) and limit > 0 else self.default_gallery_limit
        resolved_limit = max(1, min(resolved_limit, 200))
        records = self.repository.list_by_user(normalized_user_id, limit=resolved_limit)

        items: list[GalleryItem] = []
        seen: set[str] = set()
        for record in records:
            if not record.image_key or record.image_key in seen:
                continue
            items.append(
                GalleryItem(
                    memory_id=record.memory_id,
                    image_key=record.image_key,
                    image_url=record.image_url,
                    captured_at=record.captured_at,
                    caption=record.caption,
                    scene_summary=record.scene_summary,
                    position_hint=record.position_hint,
                )
            )
            seen.add(record.image_key)
        return items

    def check_health(self) -> None:
        self.repository.check_health()
        self.signer.check_health()


def build_default_media_url_signer() -> S3MediaUrlSigner:
    expires_raw = os.getenv("API_MEDIA_ACCESS_URL_EXPIRES_SEC", "300").strip() or "300"
    try:
        default_expiration_sec = int(expires_raw)
    except ValueError as exc:
        raise MediaUrlSignerConfigError(
            "API_MEDIA_ACCESS_URL_EXPIRES_SEC must be an integer"
        ) from exc
    return S3MediaUrlSigner(default_expiration_sec=default_expiration_sec)


def build_default_media_access_service() -> MediaAccessService:
    database_url = os.getenv("API_CAPTURE_DATABASE_URL", "").strip()
    if not database_url:
        raise MediaUrlSignerConfigError(
            "API_CAPTURE_DATABASE_URL is required for media ownership verification"
        )
    gallery_limit_raw = os.getenv("API_MEDIA_GALLERY_DEFAULT_LIMIT", "50").strip() or "50"
    try:
        default_gallery_limit = int(gallery_limit_raw)
    except ValueError as exc:
        raise MediaUrlSignerConfigError(
            "API_MEDIA_GALLERY_DEFAULT_LIMIT must be an integer"
        ) from exc
    return MediaAccessService(
        repository=PostgresMemoryStoreClient(database_url),
        signer=build_default_media_url_signer(),
        default_gallery_limit=default_gallery_limit,
    )
