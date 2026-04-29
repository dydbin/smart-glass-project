import json
import re
from typing import Any, Dict, List


BRAND_MAP = {
    "airpods": "Apple",
    "아이폰": "Apple",
    "맥북": "Apple",
    "macbook": "Apple",
    "iphone": "Apple",
    "아이패드": "Apple",
    "ipad": "Apple",
    "magsafe": "Apple",
    "apple pencil": "Apple",
    "갤럭시": "Samsung",
    "galaxy": "Samsung",
    "버즈": "Samsung",
    "buds": "Samsung",
    "갤럭시탭": "Samsung",
}

_SPATIAL_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
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
_SPATIAL_CONNECTOR_SPLIT = re.compile(
    r"\s+(?:next to|beside|near|inside|under|on top of|on|in)\b",
    re.I,
)
_OBJECT_TOKEN_CLEANER = re.compile(r"[^0-9a-zA-Z가-힣]+")
_ALIAS_GROUPS: tuple[tuple[str, ...], ...] = (
    ("맥북", "macbook", "노트북", "laptop"),
    ("아이폰", "iphone", "스마트폰", "휴대폰", "phone"),
    ("아이패드", "ipad", "태블릿", "tablet", "터치 패드"),
    ("apple pencil", "애플펜슬", "스타일러스", "stylus pen", "stylus"),
    ("airpods 케이스", "에어팟 케이스", "airpods case"),
    ("airpods", "에어팟", "무선 이어폰"),
)
_SCENE_LIKE_NAMES = {
    "가정",
    "거실",
    "공간",
    "내부",
    "내부 공간",
    "내부 장소",
    "실내",
    "업무 공간",
    "작업 공간",
    "인테리어",
    "장면",
    "집안",
}


def _optional_json_repair():
    try:
        from json_repair import repair_json

        return repair_json
    except Exception:
        return None


def extract_json_payload(text: str, expect_array: bool = False) -> Any:
    cleaned = (text or "").replace("```json", "").replace("```", "").strip()
    start_char = "[" if expect_array else "{"
    end_char = "]" if expect_array else "}"
    start = cleaned.find(start_char)
    end = cleaned.rfind(end_char)
    if start == -1 or end == -1 or end < start:
        return [] if expect_array else {}

    candidate = cleaned[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        repair_json = _optional_json_repair()
        if repair_json is None:
            return [] if expect_array else {}
        try:
            return json.loads(repair_json(candidate))
        except Exception:
            return [] if expect_array else {}


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def is_noisy_name(name: Any) -> bool:
    normalized = _normalize_text(name)
    if not normalized:
        return True
    if normalized in _SCENE_LIKE_NAMES:
        return True
    if "_" in normalized:
        return True
    if (
        normalized == normalized.lower()
        and normalized.isascii()
        and normalized.replace(" ", "").isalpha()
    ):
        return True
    return False


def infer_brand_from_name(name: Any) -> str | None:
    normalized = _normalize_text(name).lower()
    for keyword, brand in BRAND_MAP.items():
        if keyword in normalized:
            return brand
    return None


def _object_key(name: Any) -> str:
    return _OBJECT_TOKEN_CLEANER.sub("", _normalize_text(name).lower())


def _alias_group_for_name(name: Any) -> tuple[str, ...] | None:
    object_key = _object_key(name)
    if not object_key:
        return None
    for group in _ALIAS_GROUPS:
        if any(_object_key(alias) == object_key for alias in group):
            return group
    return None


def _prefer_more_specific_name(current: str, candidate: str) -> str:
    current_group = _alias_group_for_name(current)
    candidate_group = _alias_group_for_name(candidate)
    if current_group is None or candidate_group is None or current_group != candidate_group:
        return current

    normalized_current = _normalize_text(current)
    normalized_candidate = _normalize_text(candidate)
    for alias in current_group:
        if _object_key(normalized_candidate) == alias:
            return normalized_candidate
        if _object_key(normalized_current) == alias:
            return normalized_current
    return normalized_current


def _extract_position_hint(caption: str | None) -> str | None:
    cleaned = _normalize_text(caption)
    if not cleaned:
        return None

    for pattern, template in _SPATIAL_RULES:
        match = pattern.search(cleaned)
        if not match:
            continue
        target = _normalize_text(match.group(1))
        target = _SPATIAL_CONNECTOR_SPLIT.split(target, maxsplit=1)[0]
        target = re.sub(r"^(?:the|a|an)\s+", "", target, flags=re.I)
        target = _normalize_text(target)
        if target:
            return template.format(obj=target)

    return None


def _normalize_string_list(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []

    normalized: List[str] = []
    seen: set[str] = set()
    for value in values:
        if isinstance(value, list):
            nested_values = value
        else:
            nested_values = [value]
        for nested_value in nested_values:
            if isinstance(nested_value, dict):
                continue
            text = _normalize_text(nested_value)
            if not text or text in seen:
                continue
            normalized.append(text)
            seen.add(text)
    return normalized


def deduplicate_object_names(values: List[Any]) -> List[str]:
    deduped: List[str] = []
    seen_exact: set[str] = set()
    alias_index: Dict[tuple[str, ...], int] = {}

    for value in values:
        text = _normalize_text(value)
        if not text or is_noisy_name(text):
            continue
        if text in seen_exact:
            continue

        alias_group = _alias_group_for_name(text)
        if alias_group is None:
            deduped.append(text)
            seen_exact.add(text)
            continue

        existing_index = alias_index.get(alias_group)
        if existing_index is None:
            deduped.append(text)
            seen_exact.add(text)
            alias_index[alias_group] = len(deduped) - 1
            continue

        preferred = _prefer_more_specific_name(deduped[existing_index], text)
        deduped[existing_index] = preferred
        seen_exact.add(text)

    return deduped


def _normalize_brand(value: Any, *, fallback_name: Any = None) -> str | None:
    if value is None or value == {} or value == []:
        return infer_brand_from_name(fallback_name)
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, list) and value:
        return _normalize_text(value[0]) or infer_brand_from_name(fallback_name)
    if isinstance(value, dict) and value:
        return _normalize_text(next(iter(value.keys()))) or infer_brand_from_name(
            fallback_name
        )
    return infer_brand_from_name(fallback_name)


def _normalize_position(value: Any) -> Dict[str, str]:
    if not isinstance(value, dict):
        return {"depth_hint": "unknown", "surface": "unknown"}
    clean: Dict[str, str] = {}
    for key in ("depth_hint", "surface"):
        text = _normalize_text(value.get(key))
        clean[key] = text or "unknown"
    return clean


def _normalize_nearby_objects(value: Any) -> List[str]:
    result = []
    seen: set[str] = set()
    for item in _normalize_string_list(value):
        if is_noisy_name(item) or item in seen:
            continue
        result.append(item)
        seen.add(item)
    return result


def _estimate_confidence(record: Dict[str, Any]) -> float:
    score = 0.5
    surface = record.get("position", {}).get("surface", "")
    if surface and surface not in {"unknown", "테이블 위"}:
        score += 0.2
    if record.get("visual_features", {}).get("brand"):
        score += 0.1
    if len(record.get("nearby_objects", [])) >= 2:
        score += 0.2
    return round(min(score, 1.0), 2)


def _normalize_location(value: Any) -> Dict[str, Any] | None:
    if not isinstance(value, dict):
        return None

    location = {
        "name": _normalize_text(value.get("name")) or None,
        "address": _normalize_text(value.get("address")) or None,
        "latitude": value.get("latitude"),
        "longitude": value.get("longitude"),
    }
    if not any(location.values()):
        return None
    return location


def normalize_qwen_metadata(payload: Dict[str, Any]) -> Dict[str, Any]:
    caption = _normalize_text(payload.get("caption")) or None
    scene_summary = _normalize_text(payload.get("sceneSummary")) or None
    if scene_summary is None:
        scene_summary = _normalize_text(payload.get("scene_summary")) or None

    raw_detected_objects = payload.get("detectedObjects")
    if isinstance(raw_detected_objects, list):
        detected_objects = _normalize_string_list(raw_detected_objects)
    else:
        detected_objects = []
    tags = _normalize_string_list(payload.get("tags", []))
    ocr_text = _normalize_text(payload.get("ocrText", payload.get("ocr_text"))) or None
    position_hint = _normalize_text(
        payload.get("positionHint", payload.get("position_hint"))
    ) or None
    location = _normalize_location(payload.get("location"))

    if position_hint is None and caption:
        position_hint = _extract_position_hint(caption)

    return {
        "caption": caption,
        "sceneSummary": scene_summary,
        "detectedObjects": deduplicate_object_names(detected_objects),
        "tags": tags,
        "ocrText": ocr_text,
        "positionHint": position_hint,
        "location": location,
    }


def normalize_structured_objects(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    objects = payload.get("objects", [])
    if not isinstance(objects, list):
        return []

    normalized_objects: List[Dict[str, Any]] = []
    for index, item in enumerate(objects, start=1):
        if not isinstance(item, dict):
            continue
        name = _normalize_text(item.get("name"))
        if not name or is_noisy_name(name):
            continue
        visual_features = item.get("visual_features", {})
        if not isinstance(visual_features, dict):
            visual_features = {}
        record: Dict[str, Any] = {
            "object_id": index,
            "name": name,
            "position": _normalize_position(item.get("position")),
            "nearby_objects": _normalize_nearby_objects(
                item.get("nearby_objects", item.get("nearby"))
            ),
            "visual_features": {
                "color": _normalize_text(visual_features.get("color")) or "unknown",
                "material": _normalize_text(visual_features.get("material"))
                or "unknown",
                "brand": _normalize_brand(
                    visual_features.get("brand"),
                    fallback_name=name,
                ),
                "shape": _normalize_text(visual_features.get("shape")) or "unknown",
            },
            "raw_description_ko": _normalize_text(
                item.get("raw_description_ko", item.get("description"))
            )
            or None,
        }
        record["confidence"] = _estimate_confidence(record)
        normalized_objects.append(record)
    return normalized_objects


def object_names_from_structured_payload(payload: Dict[str, Any]) -> List[str]:
    return deduplicate_object_names(
        [item["name"] for item in normalize_structured_objects(payload)]
    )


def collect_nearby_candidate_names(
    objects: List[Dict[str, Any]], detected_objects: List[Any] | None = None
) -> List[str]:
    main_names = set(deduplicate_object_names(detected_objects or []))
    main_names.update(item["name"] for item in objects if item.get("name"))
    candidates: List[str] = []
    for item in objects:
        candidates.extend(item.get("nearby_objects", []))
    return [
        name
        for name in deduplicate_object_names(candidates)
        if name not in main_names
    ]


def merge_detected_object_candidates(
    *,
    detected_objects: List[Any],
    structured_objects: List[Dict[str, Any]],
) -> List[str]:
    merged = list(detected_objects or [])
    merged.extend(item["name"] for item in structured_objects if item.get("name"))
    merged.extend(collect_nearby_candidate_names(structured_objects, detected_objects))
    return deduplicate_object_names(merged)


def tags_from_korean_text(*values: Any) -> List[str]:
    tags: List[str] = []
    seen: set[str] = set()
    for value in values:
        text = _normalize_text(value)
        if not text:
            continue
        for token in re.split(r"[,/|]", text):
            normalized = _normalize_text(token)
            if len(normalized) < 2 or normalized in seen:
                continue
            tags.append(normalized)
            seen.add(normalized)
    return tags


def build_metadata_tags(
    metadata: Dict[str, Any], structured_objects: List[Dict[str, Any]]
) -> List[str]:
    object_names = [item["name"] for item in structured_objects]
    nearby_names: List[str] = []
    brands: List[str] = []
    for item in structured_objects:
        nearby_names.extend(item.get("nearby_objects", []))
        brand = item.get("visual_features", {}).get("brand")
        if brand:
            brands.append(brand)

    return tags_from_korean_text(
        metadata.get("positionHint"),
        *(metadata.get("tags") or []),
        *(metadata.get("detectedObjects") or []),
        *object_names,
        *nearby_names,
        *brands,
    )
