import os
from dataclasses import dataclass
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError


class StorageError(RuntimeError):
    """Base error for storage boundary failures."""


class StorageConfigError(ValueError, StorageError):
    """Raised when required storage configuration is missing."""


class StorageAccessError(StorageError):
    """Raised when the storage backend cannot be reached or accessed."""


class StorageNotFoundError(FileNotFoundError, StorageError):
    """Raised when a storage object does not exist."""


@dataclass(frozen=True, slots=True)
class StorageObject:
    bucket_name: str
    key: str
    body: bytes
    content_type: str | None = None


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise StorageConfigError(f"Missing required environment variable: {name}")
    return value


def _require_any_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    raise StorageConfigError(
        f"Missing required environment variable: one of {', '.join(names)}"
    )


def _resolve_region_name() -> str:
    return (
        os.getenv("STORAGE_REGION")
        or os.getenv("AWS_REGION")
        or "ap-northeast-2"
    ).strip() or "ap-northeast-2"


def _resolve_endpoint_url() -> str | None:
    endpoint_url = os.getenv("STORAGE_ENDPOINT_URL", "").strip()
    return endpoint_url or None


def _resolve_addressing_style() -> str:
    return (os.getenv("STORAGE_ADDRESSING_STYLE", "auto").strip().lower() or "auto")


def _resolve_bucket_name(bucket_name: str | None = None) -> str:
    normalized = (
        bucket_name
        or os.getenv("STORAGE_BUCKET_NAME")
        or os.getenv("AWS_S3_BUCKET_NAME")
        or ""
    ).strip()
    if not normalized:
        raise StorageConfigError(
            "Missing required environment variable: STORAGE_BUCKET_NAME"
        )
    return normalized


def _normalize_object_key(key: str) -> str:
    if key is None:
        raise ValueError("storage object key must not be blank")

    normalized = str(key).strip()
    if not normalized:
        raise ValueError("storage object key must not be blank")
    if normalized.startswith("/"):
        raise ValueError("storage object key must be relative")

    segments = normalized.split("/")
    if any(segment in {".", ".."} for segment in segments):
        raise ValueError("storage object key must not contain path traversal segments")
    return normalized


def _build_s3_client():
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
        client_kwargs["config"] = Config(s3={"addressing_style": addressing_style})

    return boto3.client(
        **client_kwargs,
    )


def _format_storage_message(action: str, bucket_name: str, key: str, error: Exception) -> str:
    return f"Failed to {action} storage object '{key}' in bucket '{bucket_name}': {error}"


def _format_bucket_message(action: str, bucket_name: str, error: Exception) -> str:
    return f"Failed to {action} storage bucket '{bucket_name}': {error}"


def _translate_client_error(
    *,
    action: str,
    bucket_name: str,
    key: str,
    error: Exception,
) -> StorageError:
    message = _format_storage_message(action, bucket_name, key, error)

    if isinstance(error, ClientError):
        error_code = (
            error.response.get("Error", {}).get("Code", "")
            if isinstance(error.response, dict)
            else ""
        )
        if error_code in {"404", "NoSuchKey", "NotFound"}:
            return StorageNotFoundError(message)
        return StorageAccessError(message)

    if isinstance(error, BotoCoreError):
        return StorageAccessError(message)

    if isinstance(error, StorageError):
        return error

    return StorageAccessError(message)


def _translate_bucket_error(
    *,
    action: str,
    bucket_name: str,
    error: Exception,
) -> StorageError:
    message = _format_bucket_message(action, bucket_name, error)

    if isinstance(error, ClientError):
        return StorageAccessError(message)

    if isinstance(error, BotoCoreError):
        return StorageAccessError(message)

    if isinstance(error, StorageError):
        return error

    return StorageAccessError(message)


class S3StorageService:
    def __init__(
        self,
        *,
        client: Any | None = None,
        default_bucket_name: str | None = None,
    ) -> None:
        self._client = client
        self._default_bucket_name = default_bucket_name

    def _get_client(self):
        if self._client is None:
            self._client = _build_s3_client()
        return self._client

    def _resolve_bucket_name(self, bucket_name: str | None = None) -> str:
        return _resolve_bucket_name(bucket_name or self._default_bucket_name)

    def read_object(
        self,
        key: str,
        *,
        bucket_name: str | None = None,
    ) -> StorageObject:
        normalized_key = _normalize_object_key(key)
        resolved_bucket_name = self._resolve_bucket_name(bucket_name)
        try:
            response = self._get_client().get_object(
                Bucket=resolved_bucket_name,
                Key=normalized_key,
            )
        except (ClientError, BotoCoreError, StorageError) as exc:
            raise _translate_client_error(
                action="read",
                bucket_name=resolved_bucket_name,
                key=normalized_key,
                error=exc,
            ) from exc

        return StorageObject(
            bucket_name=resolved_bucket_name,
            key=normalized_key,
            body=response["Body"].read(),
            content_type=response.get("ContentType"),
        )

    def write_object(
        self,
        key: str,
        body: bytes,
        *,
        content_type: str | None = None,
        bucket_name: str | None = None,
    ) -> None:
        normalized_key = _normalize_object_key(key)
        resolved_bucket_name = self._resolve_bucket_name(bucket_name)
        put_kwargs = {
            "Bucket": resolved_bucket_name,
            "Key": normalized_key,
            "Body": body,
        }
        if content_type:
            put_kwargs["ContentType"] = content_type

        try:
            self._get_client().put_object(**put_kwargs)
        except (ClientError, BotoCoreError, StorageError) as exc:
            raise _translate_client_error(
                action="write",
                bucket_name=resolved_bucket_name,
                key=normalized_key,
                error=exc,
            ) from exc

    def probe_bucket_access(self, *, bucket_name: str | None = None) -> None:
        resolved_bucket_name = self._resolve_bucket_name(bucket_name)
        try:
            self._get_client().head_bucket(Bucket=resolved_bucket_name)
        except (ClientError, BotoCoreError, StorageError) as exc:
            raise _translate_bucket_error(
                action="probe",
                bucket_name=resolved_bucket_name,
                error=exc,
            ) from exc


_storage_service: S3StorageService | None = None


def get_storage_service() -> S3StorageService:
    global _storage_service
    if _storage_service is None:
        _storage_service = S3StorageService()
    return _storage_service


def get_s3_client():
    return get_storage_service()._get_client()


class _LazyS3Client:
    def __getattr__(self, item):
        return getattr(get_s3_client(), item)


# Backward-compatible import for any untouched call sites.
s3_client = _LazyS3Client()
