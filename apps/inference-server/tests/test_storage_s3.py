import os
import unittest
from unittest.mock import patch

from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from src.storage.s3 import (
    S3StorageService,
    StorageAccessError,
    StorageConfigError,
    StorageNotFoundError,
)


class _FakeBody:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self) -> bytes:
        return self._payload


class _FakeS3Client:
    def __init__(self, *, payload: bytes = b"image-bytes", content_type: str = "image/jpeg"):
        self.payload = payload
        self.content_type = content_type
        self.get_calls: list[dict[str, object]] = []
        self.put_calls: list[dict[str, object]] = []

    def get_object(self, Bucket: str, Key: str) -> dict[str, object]:
        self.get_calls.append({"Bucket": Bucket, "Key": Key})
        return {
            "Body": _FakeBody(self.payload),
            "ContentType": self.content_type,
        }

    def put_object(self, **kwargs) -> None:
        self.put_calls.append(kwargs)


class _MissingObjectClient:
    def get_object(self, Bucket: str, Key: str) -> dict[str, object]:
        raise ClientError(
            {
                "Error": {
                    "Code": "NoSuchKey",
                    "Message": "missing object",
                }
            },
            "GetObject",
        )


class _AccessDeniedClient:
    def put_object(self, **kwargs) -> None:
        raise ClientError(
            {
                "Error": {
                    "Code": "AccessDenied",
                    "Message": "denied",
                }
            },
            "PutObject",
        )

    def head_bucket(self, Bucket: str) -> None:
        raise ClientError(
            {
                "Error": {
                    "Code": "AccessDenied",
                    "Message": "denied",
                }
            },
            "HeadBucket",
        )


class _BrokenClient:
    def get_object(self, Bucket: str, Key: str) -> dict[str, object]:
        raise BotoCoreError()

    def head_bucket(self, Bucket: str) -> None:
        raise BotoCoreError()


class _ProbeClient:
    def __init__(self) -> None:
        self.bucket_calls: list[str] = []

    def head_bucket(self, Bucket: str) -> None:
        self.bucket_calls.append(Bucket)


class StorageServiceTestCase(unittest.TestCase):
    def tearDown(self) -> None:
        for name in (
            "STORAGE_ACCESS_KEY_ID",
            "STORAGE_SECRET_ACCESS_KEY",
            "STORAGE_REGION",
            "STORAGE_BUCKET_NAME",
            "STORAGE_ENDPOINT_URL",
            "STORAGE_ADDRESSING_STYLE",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_REGION",
            "AWS_S3_BUCKET_NAME",
        ):
            os.environ.pop(name, None)

    def test_read_object_returns_storage_object(self) -> None:
        service = S3StorageService(
            client=_FakeS3Client(payload=b"abc123", content_type="image/png"),
            default_bucket_name="smart-glass-test",
        )

        result = service.read_object("captures/test.png")

        self.assertEqual(result.bucket_name, "smart-glass-test")
        self.assertEqual(result.key, "captures/test.png")
        self.assertEqual(result.body, b"abc123")
        self.assertEqual(result.content_type, "image/png")

    def test_read_object_normalizes_surrounding_whitespace(self) -> None:
        client = _FakeS3Client(payload=b"abc123", content_type="image/png")
        service = S3StorageService(
            client=client,
            default_bucket_name="smart-glass-test",
        )

        result = service.read_object("  captures/test.png  ")

        self.assertEqual(result.key, "captures/test.png")
        self.assertEqual(client.get_calls[0]["Key"], "captures/test.png")

    def test_read_object_rejects_blank_object_key(self) -> None:
        service = S3StorageService(
            client=_FakeS3Client(),
            default_bucket_name="smart-glass-test",
        )

        with self.assertRaisesRegex(ValueError, "must not be blank"):
            service.read_object("   ")

    def test_read_object_rejects_absolute_object_key(self) -> None:
        service = S3StorageService(
            client=_FakeS3Client(),
            default_bucket_name="smart-glass-test",
        )

        with self.assertRaisesRegex(ValueError, "must be relative"):
            service.read_object("/captures/test.png")

    def test_read_object_rejects_path_traversal_object_key(self) -> None:
        service = S3StorageService(
            client=_FakeS3Client(),
            default_bucket_name="smart-glass-test",
        )

        with self.assertRaisesRegex(ValueError, "path traversal"):
            service.read_object("captures/../secrets.png")

    def test_write_object_passes_content_type_to_client(self) -> None:
        client = _FakeS3Client()
        service = S3StorageService(
            client=client,
            default_bucket_name="smart-glass-test",
        )

        service.write_object(
            "captures/test.png",
            b"payload",
            content_type="image/png",
        )

        self.assertEqual(len(client.put_calls), 1)
        self.assertEqual(client.put_calls[0]["Bucket"], "smart-glass-test")
        self.assertEqual(client.put_calls[0]["Key"], "captures/test.png")
        self.assertEqual(client.put_calls[0]["Body"], b"payload")
        self.assertEqual(client.put_calls[0]["ContentType"], "image/png")

    def test_write_object_normalizes_object_key(self) -> None:
        client = _FakeS3Client()
        service = S3StorageService(
            client=client,
            default_bucket_name="smart-glass-test",
        )

        service.write_object("  captures/test.png  ", b"payload")

        self.assertEqual(client.put_calls[0]["Key"], "captures/test.png")

    def test_read_object_raises_not_found_for_missing_key(self) -> None:
        service = S3StorageService(
            client=_MissingObjectClient(),
            default_bucket_name="smart-glass-test",
        )

        with self.assertRaises(StorageNotFoundError):
            service.read_object("captures/missing.png")

    def test_write_object_raises_access_error_for_denied_client_error(self) -> None:
        service = S3StorageService(
            client=_AccessDeniedClient(),
            default_bucket_name="smart-glass-test",
        )

        with self.assertRaises(StorageAccessError):
            service.write_object("captures/test.png", b"payload")

    def test_read_object_raises_access_error_for_botocore_failure(self) -> None:
        service = S3StorageService(
            client=_BrokenClient(),
            default_bucket_name="smart-glass-test",
        )

        with self.assertRaises(StorageAccessError):
            service.read_object("captures/test.png")

    def test_service_raises_config_error_when_bucket_is_missing(self) -> None:
        service = S3StorageService(client=_FakeS3Client())

        with self.assertRaises(StorageConfigError):
            service.read_object("captures/test.png")

    def test_service_builds_client_from_env(self) -> None:
        os.environ["STORAGE_ACCESS_KEY_ID"] = "test-access"
        os.environ["STORAGE_SECRET_ACCESS_KEY"] = "test-secret"
        os.environ["STORAGE_REGION"] = "kr-standard"
        os.environ["STORAGE_BUCKET_NAME"] = "smart-glass-test"
        os.environ["STORAGE_ENDPOINT_URL"] = "https://kr.object.ncloudstorage.com"
        os.environ["STORAGE_ADDRESSING_STYLE"] = "path"

        fake_client = _FakeS3Client()
        with patch("src.storage.s3.boto3.client", return_value=fake_client) as mocked_client:
            service = S3StorageService()
            service.read_object("captures/test.png")

        mocked_client.assert_called_once()
        _, kwargs = mocked_client.call_args
        self.assertEqual(kwargs["service_name"], "s3")
        self.assertEqual(kwargs["aws_access_key_id"], "test-access")
        self.assertEqual(kwargs["aws_secret_access_key"], "test-secret")
        self.assertEqual(kwargs["region_name"], "kr-standard")
        self.assertEqual(kwargs["endpoint_url"], "https://kr.object.ncloudstorage.com")
        self.assertIsInstance(kwargs["config"], Config)
        self.assertEqual(kwargs["config"].s3.get("addressing_style"), "path")

    def test_probe_bucket_access_calls_head_bucket(self) -> None:
        client = _ProbeClient()
        service = S3StorageService(
            client=client,
            default_bucket_name="smart-glass-test",
        )

        service.probe_bucket_access()

        self.assertEqual(client.bucket_calls, ["smart-glass-test"])

    def test_probe_bucket_access_raises_access_error(self) -> None:
        service = S3StorageService(
            client=_AccessDeniedClient(),
            default_bucket_name="smart-glass-test",
        )

        with self.assertRaises(StorageAccessError):
            service.probe_bucket_access()


if __name__ == "__main__":
    unittest.main()
