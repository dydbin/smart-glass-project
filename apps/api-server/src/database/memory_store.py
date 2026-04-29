from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Mapping

try:  # pragma: no cover - optional runtime dependency
    import psycopg
except ImportError:  # pragma: no cover - handled at runtime
    psycopg = None


SPATIAL_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"\bnext to\s+(?:the\s+|a\s+|an\s+)?([a-z0-9][a-z0-9\s-]{1,40})",
            re.I,
        ),
        "{obj} 옆",
    ),
    (
        re.compile(
            r"\bbeside\s+(?:the\s+|a\s+|an\s+)?([a-z0-9][a-z0-9\s-]{1,40})",
            re.I,
        ),
        "{obj} 옆",
    ),
    (
        re.compile(
            r"\bnear\s+(?:the\s+|a\s+|an\s+)?([a-z0-9][a-z0-9\s-]{1,40})",
            re.I,
        ),
        "{obj} 근처",
    ),
    (
        re.compile(
            r"\binside\s+(?:the\s+|a\s+|an\s+)?([a-z0-9][a-z0-9\s-]{1,40})",
            re.I,
        ),
        "{obj} 안",
    ),
    (
        re.compile(
            r"\bunder\s+(?:the\s+|a\s+|an\s+)?([a-z0-9][a-z0-9\s-]{1,40})",
            re.I,
        ),
        "{obj} 아래",
    ),
    (
        re.compile(
            r"\bon top of\s+(?:the\s+|a\s+|an\s+)?([a-z0-9][a-z0-9\s-]{1,40})",
            re.I,
        ),
        "{obj} 위",
    ),
    (
        re.compile(
            r"\bon\s+(?:the\s+|a\s+|an\s+)?([a-z0-9][a-z0-9\s-]{1,40})",
            re.I,
        ),
        "{obj} 위",
    ),
    (
        re.compile(
            r"\bin\s+(?:the\s+|a\s+|an\s+)?([a-z0-9][a-z0-9\s-]{1,40})",
            re.I,
        ),
        "{obj} 안",
    ),
)

SPATIAL_CONNECTOR_SPLIT = re.compile(
    r"\s+(?:next to|beside|near|inside|under|on top of|on|in)\b",
    re.I,
)


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def _normalize_timestamp(value: Any, *, field_name: str) -> str | None:
    normalized = _normalize_text(value)
    if not normalized:
        return None

    candidate = normalized.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a valid ISO 8601 timestamp") from exc

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return (
        parsed.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _dedupe_strings(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []

    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _normalize_text(value)
        if not text:
            continue
        lowered = text.lower()
        if lowered in seen:
            continue
        deduped.append(text)
        seen.add(lowered)
    return deduped


def _clean_spatial_object(value: str) -> str:
    cleaned = _normalize_text(value)
    if not cleaned:
        return ""

    cleaned = SPATIAL_CONNECTOR_SPLIT.split(cleaned, maxsplit=1)[0]
    cleaned = re.sub(r"^(?:the|a|an)\s+", "", cleaned, flags=re.I)
    return _normalize_text(cleaned)


def _extract_position_hint(*texts: Any) -> str | None:
    for text in texts:
        cleaned = _normalize_text(text)
        if not cleaned:
            continue
        for pattern, template in SPATIAL_RULES:
            match = pattern.search(cleaned)
            if not match:
                continue
            spatial_object = _clean_spatial_object(match.group(1))
            if spatial_object:
                return template.format(obj=spatial_object)
    return None


def _normalize_structured_objects(values: Any) -> list[dict[str, Any]]:
    if not isinstance(values, list):
        return []

    normalized: list[dict[str, Any]] = []
    for item in values:
        if not isinstance(item, dict):
            continue
        name = _normalize_text(item.get("name"))
        if not name:
            continue
        visual_features = item.get("visual_features")
        brand = None
        if isinstance(visual_features, dict):
            brand = _normalize_text(visual_features.get("brand")) or None
        nearby_objects = _dedupe_strings(
            item.get("nearby_objects") or item.get("nearby") or []
        )
        normalized.append(
            {
                "name": name,
                "nearby_objects": nearby_objects,
                "visual_features": {"brand": brand},
            }
        )
    return normalized


def _build_tags(
    metadata: Mapping[str, Any], structured_objects: list[dict[str, Any]]
) -> list[str]:
    candidates: list[str] = []
    candidates.extend(_dedupe_strings(metadata.get("tags") or []))
    candidates.extend(_dedupe_strings(metadata.get("detectedObjects") or []))
    candidates.extend(item["name"] for item in structured_objects if item.get("name"))

    for item in structured_objects:
        candidates.extend(item.get("nearby_objects") or [])
        brand = item.get("visual_features", {}).get("brand")
        if brand:
            candidates.append(brand)

    tags: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        text = _normalize_text(candidate)
        if not text:
            continue
        lowered = text.lower()
        if lowered in seen:
            continue
        tags.append(text)
        seen.add(lowered)
    return tags


def _resolve_capabilities(result: Mapping[str, Any]) -> dict[str, bool] | None:
    provider_metadata = result.get("providerMetadata")
    if not isinstance(provider_metadata, Mapping):
        return None

    raw_capabilities = provider_metadata.get("capabilities")
    if not isinstance(raw_capabilities, Mapping):
        return None

    return {
        "detectedObjects": bool(raw_capabilities.get("detectedObjects", True)),
        "tags": bool(raw_capabilities.get("tags", True)),
        "positionHint": bool(raw_capabilities.get("positionHint", True)),
        "sceneSummary": bool(raw_capabilities.get("sceneSummary", False)),
        "ocrText": bool(raw_capabilities.get("ocrText", False)),
        "location": bool(raw_capabilities.get("location", False)),
    }


@dataclass(frozen=True, slots=True)
class MemoryLocation:
    name: str | None = None
    address: str | None = None
    latitude: float | None = None
    longitude: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "address": self.address,
            "latitude": self.latitude,
            "longitude": self.longitude,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "MemoryLocation":
        payload = payload or {}
        return cls(
            name=_normalize_text(payload.get("name")) or None,
            address=_normalize_text(payload.get("address")) or None,
            latitude=payload.get("latitude"),
            longitude=payload.get("longitude"),
        )


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    memory_id: str
    user_id: str
    image_key: str | None
    image_url: str | None
    captured_at: str | None
    caption: str | None
    scene_summary: str | None
    detected_objects: list[str]
    tags: list[str]
    ocr_text: str | None
    note: str | None
    position_hint: str | None
    location: MemoryLocation

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "user_id": self.user_id,
            "image_key": self.image_key,
            "image_url": self.image_url,
            "captured_at": self.captured_at,
            "caption": self.caption,
            "scene_summary": self.scene_summary,
            "detected_objects": self.detected_objects,
            "tags": self.tags,
            "ocr_text": self.ocr_text,
            "note": self.note,
            "position_hint": self.position_hint,
            "location": self.location.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MemoryRecord":
        return cls(
            memory_id=_normalize_text(payload.get("memory_id")),
            user_id=_normalize_text(payload.get("user_id")),
            image_key=_normalize_text(payload.get("image_key")) or None,
            image_url=_normalize_text(payload.get("image_url")) or None,
            captured_at=_normalize_timestamp(
                payload.get("captured_at"),
                field_name="captured_at",
            ),
            caption=_normalize_text(payload.get("caption")) or None,
            scene_summary=_normalize_text(payload.get("scene_summary")) or None,
            detected_objects=_dedupe_strings(payload.get("detected_objects") or []),
            tags=_dedupe_strings(payload.get("tags") or []),
            ocr_text=_normalize_text(payload.get("ocr_text")) or None,
            note=_normalize_text(payload.get("note")) or None,
            position_hint=_normalize_text(payload.get("position_hint")) or None,
            location=MemoryLocation.from_dict(payload.get("location")),
        )


@dataclass(frozen=True, slots=True)
class MemoryStoreOutcome:
    stored_count: int
    total_user_memories: dict[str, int]


class MemoryStoreUnavailableError(RuntimeError):
    pass


def _build_location(
    metadata: Mapping[str, Any],
    pipeline_output: Mapping[str, Any],
    capabilities: Mapping[str, bool] | None = None,
) -> MemoryLocation:
    if capabilities is not None and not capabilities.get("location"):
        return MemoryLocation()

    location = metadata.get("location")
    if isinstance(location, Mapping):
        payload = MemoryLocation(
            name=_normalize_text(location.get("name")) or None,
            address=_normalize_text(location.get("address")) or None,
            latitude=location.get("latitude"),
            longitude=location.get("longitude"),
        )
        if any([payload.name, payload.address, payload.latitude, payload.longitude]):
            return payload

    location_context = _normalize_text(pipeline_output.get("location_context"))
    if location_context and (capabilities is None or capabilities.get("location")):
        return MemoryLocation(name=location_context)
    return MemoryLocation()


def memory_record_from_vlm_result(result: Mapping[str, Any]) -> MemoryRecord:
    status = _normalize_text(result.get("status")).lower()
    if status != "success":
        raise ValueError("Only successful VLM inference results can be stored")

    memory_id = _normalize_text(result.get("memoryId"))
    user_id = _normalize_text(result.get("userId"))
    if not memory_id or not user_id:
        raise ValueError("memoryId and userId are required to store VLM results")

    source_image = result.get("sourceImage") or {}
    metadata = result.get("metadata") or {}
    pipeline_output = result.get("pipelineOutput") or {}
    capabilities = _resolve_capabilities(result)

    if capabilities is None:
        structured_objects = _normalize_structured_objects(
            pipeline_output.get("objects") or []
        )

        detected_objects = _dedupe_strings(metadata.get("detectedObjects") or [])
        if not detected_objects and structured_objects:
            detected_objects = [
                item["name"] for item in structured_objects if item.get("name")
            ]

        tags = _build_tags(metadata, structured_objects)

        scene_summary = (
            _normalize_text(metadata.get("sceneSummary"))
            or _normalize_text(pipeline_output.get("scene_summary"))
            or _normalize_text(metadata.get("caption"))
            or None
        )
        ocr_text = (
            _normalize_text(metadata.get("ocrText"))
            or _normalize_text(pipeline_output.get("ocr_text"))
            or None
        )
        position_hint = (
            _normalize_text(metadata.get("positionHint"))
            or _normalize_text(pipeline_output.get("location_context"))
            or _extract_position_hint(
                metadata.get("caption"),
                metadata.get("sceneSummary"),
                pipeline_output.get("scene_summary"),
            )
            or None
        )
        location = _build_location(metadata, pipeline_output)
    else:
        detected_objects = (
            _dedupe_strings(metadata.get("detectedObjects") or [])
            if capabilities["detectedObjects"]
            else []
        )
        tags = (
            _dedupe_strings(metadata.get("tags") or [])
            if capabilities["tags"]
            else []
        )
        scene_summary = (
            _normalize_text(metadata.get("sceneSummary"))
            if capabilities["sceneSummary"]
            else None
        ) or None
        ocr_text = (
            _normalize_text(metadata.get("ocrText"))
            if capabilities["ocrText"]
            else None
        ) or None
        position_hint = (
            _normalize_text(metadata.get("positionHint"))
            if capabilities["positionHint"]
            else None
        ) or None
        if position_hint is None:
            position_hint = _extract_position_hint(
                metadata.get("caption"),
                metadata.get("sceneSummary"),
            )
        location = _build_location(metadata, pipeline_output, capabilities)

    return MemoryRecord(
        memory_id=memory_id,
        user_id=user_id,
        image_key=_normalize_text(source_image.get("imageKey")) or None,
        image_url=_normalize_text(source_image.get("imageUrl")) or None,
        captured_at=_normalize_timestamp(
            result.get("capturedAt"),
            field_name="capturedAt",
        ),
        caption=_normalize_text(metadata.get("caption")) or None,
        scene_summary=scene_summary,
        detected_objects=detected_objects,
        tags=tags,
        ocr_text=ocr_text,
        note=None,
        position_hint=position_hint,
        location=location,
    )


class PostgresMemoryStoreClient:
    backend_name = "postgres"
    table_name = "memory_records"

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url.strip()
        if not self.database_url:
            raise ValueError("database_url must not be blank")
        self._lock = Lock()
        self._schema_ready = False

    def _require_driver(self) -> None:
        if psycopg is None:
            raise RuntimeError(
                "psycopg is required for api-server memory storage. "
                "Install apps/api-server requirements."
            )

    def _connect(self):  # type: ignore[no-untyped-def]
        self._require_driver()
        try:
            return psycopg.connect(self.database_url)
        except Exception as exc:
            raise MemoryStoreUnavailableError(
                f"Postgres connection failed: {exc}"
            ) from exc

    def _ensure_schema(self) -> None:
        if self._schema_ready:
            return

        with self._lock:
            if self._schema_ready:
                return

            try:
                with self._connect() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            f"""
                            CREATE TABLE IF NOT EXISTS {self.table_name} (
                                user_id TEXT NOT NULL,
                                memory_id TEXT NOT NULL,
                                captured_at TEXT,
                                document JSONB NOT NULL,
                                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                                PRIMARY KEY (user_id, memory_id)
                            )
                            """
                        )
                        cur.execute(
                            f"""
                            CREATE INDEX IF NOT EXISTS idx_{self.table_name}_user_id
                            ON {self.table_name} (user_id)
                            """
                        )
                        cur.execute(
                            f"""
                            CREATE INDEX IF NOT EXISTS idx_{self.table_name}_user_id_captured_at
                            ON {self.table_name} (user_id, captured_at DESC, memory_id DESC)
                            """
                        )
                        cur.execute(
                            f"""
                            CREATE INDEX IF NOT EXISTS idx_{self.table_name}_user_id_image_key
                            ON {self.table_name} (user_id, ((document->>'image_key')))
                            """
                        )
                    conn.commit()
            except MemoryStoreUnavailableError:
                raise
            except Exception as exc:
                raise MemoryStoreUnavailableError(
                    f"Postgres schema initialization failed: {exc}"
                ) from exc

            self._schema_ready = True

    def persist_vlm_result(self, worker_result: Mapping[str, Any]) -> MemoryStoreOutcome:
        record = memory_record_from_vlm_result(worker_result)
        self._ensure_schema()

        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        INSERT INTO {self.table_name} (
                            user_id,
                            memory_id,
                            captured_at,
                            document,
                            created_at,
                            updated_at
                        ) VALUES (%s, %s, %s, %s::jsonb, NOW(), NOW())
                        ON CONFLICT (user_id, memory_id) DO UPDATE SET
                            captured_at = EXCLUDED.captured_at,
                            document = EXCLUDED.document,
                            updated_at = NOW()
                        """,
                        (
                            record.user_id,
                            record.memory_id,
                            record.captured_at,
                            json.dumps(record.to_dict(), ensure_ascii=False),
                        ),
                    )
                    cur.execute(
                        f"SELECT COUNT(*) FROM {self.table_name} WHERE user_id = %s",
                        (record.user_id,),
                    )
                    row = cur.fetchone()
                conn.commit()
        except MemoryStoreUnavailableError:
            raise
        except Exception as exc:
            raise MemoryStoreUnavailableError(
                f"Postgres write failed: {exc}"
            ) from exc

        return MemoryStoreOutcome(
            stored_count=1,
            total_user_memories={record.user_id: int(row[0] if row else 0)},
        )

    def list_by_user(
        self,
        user_id: str,
        *,
        limit: int | None = None,
    ) -> list[MemoryRecord]:
        normalized_user_id = _normalize_text(user_id)
        if not normalized_user_id:
            return []

        self._ensure_schema()
        query = (
            f"""
            SELECT document::text
            FROM {self.table_name}
            WHERE user_id = %s
            ORDER BY (captured_at IS NULL) ASC, captured_at DESC, memory_id DESC
            """
        )
        params: tuple[Any, ...]
        if isinstance(limit, int) and limit > 0:
            query += " LIMIT %s"
            params = (normalized_user_id, limit)
        else:
            params = (normalized_user_id,)

        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                    rows = cur.fetchall()
        except MemoryStoreUnavailableError:
            raise
        except Exception as exc:
            raise MemoryStoreUnavailableError(
                f"Postgres read failed: {exc}"
            ) from exc

        documents: list[MemoryRecord] = []
        for (document_json,) in rows:
            documents.append(MemoryRecord.from_dict(json.loads(document_json)))
        return documents

    def get_by_image_key(self, user_id: str, image_key: str) -> MemoryRecord | None:
        normalized_user_id = _normalize_text(user_id)
        normalized_image_key = _normalize_text(image_key)
        if not normalized_user_id or not normalized_image_key:
            return None

        self._ensure_schema()
        query = f"""
            SELECT document::text
            FROM {self.table_name}
            WHERE user_id = %s
              AND document->>'image_key' = %s
            ORDER BY updated_at DESC
            LIMIT 1
        """

        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (normalized_user_id, normalized_image_key))
                    row = cur.fetchone()
        except MemoryStoreUnavailableError:
            raise
        except Exception as exc:
            raise MemoryStoreUnavailableError(
                f"Postgres read failed: {exc}"
            ) from exc

        if not row:
            return None
        return MemoryRecord.from_dict(json.loads(row[0]))

    def list_by_image_keys(
        self,
        user_id: str,
        image_keys: list[str],
    ) -> list[MemoryRecord]:
        normalized_user_id = _normalize_text(user_id)
        normalized_image_keys = [
            normalized
            for normalized in (_normalize_text(image_key) for image_key in image_keys)
            if normalized
        ]
        if not normalized_user_id or not normalized_image_keys:
            return []

        self._ensure_schema()
        query = f"""
            SELECT document::text
            FROM {self.table_name}
            WHERE user_id = %s
              AND document->>'image_key' = ANY(%s)
            ORDER BY updated_at DESC
        """

        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (normalized_user_id, normalized_image_keys))
                    rows = cur.fetchall()
        except MemoryStoreUnavailableError:
            raise
        except Exception as exc:
            raise MemoryStoreUnavailableError(
                f"Postgres read failed: {exc}"
            ) from exc

        return [MemoryRecord.from_dict(json.loads(document_json)) for (document_json,) in rows]

    def check_health(self) -> None:
        if not self._schema_ready:
            self._ensure_schema()
            return

        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
        except MemoryStoreUnavailableError:
            raise
        except Exception as exc:
            raise MemoryStoreUnavailableError(
                f"Postgres health check failed: {exc}"
            ) from exc
