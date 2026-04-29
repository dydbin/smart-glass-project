import math
import json
import os
import time
from datetime import datetime
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig

from src.contracts.qwen_parser import (
    build_metadata_tags,
    collect_nearby_candidate_names,
    deduplicate_object_names,
    extract_json_payload,
    normalize_structured_objects,
)


DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Always respond in Korean. "
    "Return valid JSON only without markdown code fences."
)
SYSTEM_PROMPT_LIST = DEFAULT_SYSTEM_PROMPT
USER_PROMPT_LIST = """이미지에서 보이는 물체를 빠짐없이 모두 나열하세요.

규칙:
- 작거나 부분만 보여도 포함
- 브랜드 제품은 브랜드명 포함 (예: AirPods, 맥북, 아이패드, Apple Pencil)
- 태블릿/스마트패드/아이패드도 반드시 포함
- 같은 종류가 여러 개면 모두 포함 (예: AirPods 케이스가 2개면 "AirPods 케이스 1", "AirPods 케이스 2")
- 가구/벽/바닥 제외
- JSON 배열만 출력

["물체1", "물체2", ...]"""
SYSTEM_PROMPT_DETAIL = DEFAULT_SYSTEM_PROMPT
USER_PROMPT_DETAIL = """이미지에서 "{object_name}" 하나에 대해서만 아래 JSON 형식으로 출력하세요.
설명 없이 JSON만 출력하세요.

{{
  "name": "{object_name}",
  "confidence": 0.0,
  "position": {{
    "depth_hint": "near/mid/far",
    "surface": "놓인 표면 또는 기준 물체. 예: 맥북 위, 아이폰 왼쪽 테이블, 테이블 위 왼쪽"
  }},
  "visual_features": {{
    "color": "색상",
    "material": "재질",
    "brand": null,
    "shape": "형태"
  }},
  "nearby_objects": ["주변 물체1", "주변 물체2"],
  "raw_description_ko": "한 문장 묘사"
}}

규칙:
- JSON만 출력
- confidence는 0.0~1.0 숫자
- brand는 문자열 또는 null
- nearby_objects는 문자열 배열
- surface는 절대 비워두지 말 것
- 반드시 한국어로"""
DEDUP_PROMPT_TEMPLATE = """다음은 이미지에서 감지된 물체 목록입니다:

{object_list}

규칙:
- 진짜 같은 물체를 가리키는 항목만 하나로 합칠 것 (예: "맥북"과 "노트북"이 동일한 물체면 하나만)
- 표기만 다른 동일 물체도 합칠 것 (예: "AirPods 케이스"와 "AirPods 캡"이 같은 물체면 하나만)
- 가장 구체적인 이름 사용 (예: "노트북"보다 "맥북" 선택)
- 같은 종류라도 이미지에서 위치가 다른 별개 물체면 반드시 둘 다 유지
- 확실하지 않으면 제거하지 말고 유지할 것
- 출력 이름은 반드시 위 입력 목록에 있는 이름 그대로 사용할 것 (새 이름 만들지 말 것)
- JSON 배열만 출력

["물체1", "물체2", ...]"""
MISSING_OBJECT_PROMPT_TEMPLATE = """현재 이미지에서 인식된 물체 목록입니다.

{object_list}

이미지를 다시 확인해서, 위 목록에 없는 중요한 물체가 있으면 추가한 전체 목록을 JSON 배열로 출력하세요.
- 특히 태블릿, 아이패드, 스타일러스 펜, Apple Pencil 같은 물체가 있는지 확인하세요.
- 빠진 것이 없으면 그대로 출력하세요.

["물체1", "물체2", ...]"""
SCENE_SUMMARY_PROMPT = """이미지의 전체 장면을 한 문장으로 요약하고 공간 유형을 JSON으로 출력하세요.

{
  "scene_summary": "10단어 이내 요약",
  "location_context": "공간 유형 (예: 거실, 카페, 사무실)"
}"""


@dataclass(frozen=True)
class QwenVlmSpec:
    key: str
    model_id: str
    family: str = "qwen2_5_vl"
    recommended_quantization: str = "4bit"
    max_image_pixels: int = 1280 * 1280
    notes: str = ""


QWEN_MODEL_SPECS: Dict[str, QwenVlmSpec] = {
    "qwen2.5-vl-3b": QwenVlmSpec(
        key="qwen2.5-vl-3b",
        model_id="Qwen/Qwen2.5-VL-3B-Instruct",
        max_image_pixels=1280 * 1280,
        notes="smoke test와 빠른 구조 검증용 기본 VLM.",
    ),
    "qwen2.5-vl-7b": QwenVlmSpec(
        key="qwen2.5-vl-7b",
        model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        max_image_pixels=1024 * 1024,
        notes="구조화된 JSON 응답 실험용 VLM 후보.",
    ),
}


def list_available_qwen_vlm_models() -> List[QwenVlmSpec]:
    return list(QWEN_MODEL_SPECS.values())


def get_qwen_vlm_spec(key: str) -> QwenVlmSpec:
    if key not in QWEN_MODEL_SPECS:
        raise KeyError(
            f"Unknown Qwen VLM model key: {key}. "
            f"Available keys: {', '.join(sorted(QWEN_MODEL_SPECS))}"
        )
    return QWEN_MODEL_SPECS[key]


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(
            f"Unsupported dtype: {dtype_name}. Choose one of {', '.join(mapping)}"
        )
    return mapping[dtype_name]


def _build_quantization_config(
    quantization: str, torch_dtype: torch.dtype
) -> Optional[BitsAndBytesConfig]:
    if quantization == "none":
        return None
    if quantization == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    if quantization == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    raise ValueError(
        f"Unsupported quantization: {quantization}. Choose from none, 8bit, 4bit"
    )


def _require_qwen_dependencies() -> Any:
    try:
        from qwen_vl_utils import process_vision_info
    except Exception as exc:
        raise RuntimeError(
            "qwen-vl-utils is required for Qwen VLM inference. "
            "Install dependencies from apps/inference-server/requirements.txt."
        ) from exc
    return process_vision_info


def _resolve_max_image_pixels(spec: QwenVlmSpec) -> int:
    raw_value = os.getenv("VISION_QWEN_MAX_IMAGE_PIXELS", "").strip()
    if not raw_value:
        return spec.max_image_pixels
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            "VISION_QWEN_MAX_IMAGE_PIXELS must be an integer when set."
        ) from exc
    if value <= 0:
        raise ValueError("VISION_QWEN_MAX_IMAGE_PIXELS must be greater than 0.")
    return value


def _downscale_image_if_needed(image: Image.Image, max_image_pixels: int) -> Image.Image:
    width, height = image.size
    current_pixels = width * height
    if current_pixels <= max_image_pixels:
        return image

    scale = math.sqrt(max_image_pixels / float(current_pixels))
    resized_width = max(1, int(width * scale))
    resized_height = max(1, int(height * scale))
    return image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)


def _calculate_sharpness(image: Image.Image) -> float:
    try:
        import cv2
        import numpy as np
    except Exception:
        return 0.0

    grayscale = np.array(image.convert("L"))
    laplacian = cv2.Laplacian(grayscale, cv2.CV_64F)
    return round(float(laplacian.var()), 2)


def _get_vram_usage() -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    return {
        "allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
        "reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 2),
        "max_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "total_gb": round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
        ),
    }


def _new_pipeline_profile() -> Dict[str, Dict[str, Any]]:
    return {
        "stage_timings_sec": {},
        "stage_call_counts": {},
    }


def _record_profiled_stage(
    profile: Dict[str, Dict[str, Any]],
    stage_name: str,
    started_at: float,
) -> None:
    elapsed_sec = max(0.0, time.perf_counter() - started_at)
    timings = profile["stage_timings_sec"]
    call_counts = profile["stage_call_counts"]
    timings[stage_name] = round(float(timings.get(stage_name, 0.0)) + elapsed_sec, 4)
    call_counts[stage_name] = int(call_counts.get(stage_name, 0)) + 1


def _run_profiled_stage(
    profile: Dict[str, Dict[str, Any]],
    stage_name: str,
    callback: Any,
) -> Any:
    stage_started_at = time.perf_counter()
    try:
        return callback()
    finally:
        _record_profiled_stage(profile, stage_name, stage_started_at)


def _build_pipeline_diagnostics(
    profile: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    timings = dict(profile["stage_timings_sec"])
    slowest_stage = None
    if timings:
        slowest_stage = max(timings, key=timings.get)

    return {
        "stage_timings_sec": timings,
        "stage_call_counts": dict(profile["stage_call_counts"]),
        "stage_total_sec": round(sum(timings.values()), 4),
        "slowest_stage": slowest_stage,
    }


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def _run_generation(
    *,
    model: Any,
    processor: Any,
    device_name: str,
    messages: List[Dict[str, Any]],
    process_vision_info: Any,
    max_new_tokens: int,
) -> str:
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    model_inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
    )
    model_inputs = model_inputs.to(device_name)

    with torch.inference_mode():
        output_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.2,
        )

    return processor.batch_decode(
        output_ids[:, model_inputs.input_ids.shape[1] :],
        skip_special_tokens=True,
    )[0]


def _infer_text(
    *,
    image: Image.Image,
    system_prompt: str,
    user_prompt: str,
    model: Any,
    processor: Any,
    device_name: str,
    process_vision_info: Any,
    max_new_tokens: int = 256,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]
    return _run_generation(
        model=model,
        processor=processor,
        device_name=device_name,
        messages=messages,
        process_vision_info=process_vision_info,
        max_new_tokens=max_new_tokens,
    )


def _normalize_location_context(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, dict):
        if not value:
            return "알 수 없음"
        nested_value = next(iter(value.values()))
        if isinstance(nested_value, list):
            return _normalize_text(nested_value[0]) or "알 수 없음"
        return _normalize_text(nested_value) or "알 수 없음"
    if isinstance(value, list):
        return _normalize_text(value[0]) or "알 수 없음"
    return "알 수 없음"


def _choose_position_hint(objects: List[Dict[str, Any]], location_context: str) -> str | None:
    for item in objects:
        surface = _normalize_text(item.get("position", {}).get("surface"))
        if surface and surface != "unknown":
            return surface
    if location_context and location_context != "알 수 없음":
        return location_context
    return None


def _check_missing_objects_with_vlm(
    *,
    image: Image.Image,
    current_names: List[str],
    model: Any,
    processor: Any,
    device_name: str,
    process_vision_info: Any,
) -> List[str]:
    clean_candidates = deduplicate_object_names(current_names)
    if not clean_candidates:
        return []

    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": MISSING_OBJECT_PROMPT_TEMPLATE.format(
                        object_list=json.dumps(clean_candidates, ensure_ascii=False)
                    ),
                },
            ],
        },
    ]
    raw_text = _run_generation(
        model=model,
        processor=processor,
        device_name=device_name,
        messages=messages,
        process_vision_info=process_vision_info,
        max_new_tokens=256,
    )
    updated = deduplicate_object_names(extract_json_payload(raw_text, expect_array=True))
    return [name for name in updated if name not in clean_candidates]


def _get_object_list(
    *,
    image: Image.Image,
    model: Any,
    processor: Any,
    device_name: str,
    process_vision_info: Any,
) -> List[str]:
    raw_text = _infer_text(
        image=image,
        system_prompt=SYSTEM_PROMPT_LIST,
        user_prompt=USER_PROMPT_LIST,
        model=model,
        processor=processor,
        device_name=device_name,
        process_vision_info=process_vision_info,
        max_new_tokens=256,
    )
    object_list = deduplicate_object_names(extract_json_payload(raw_text, expect_array=True))
    return object_list


def _get_scene_summary(
    *,
    image: Image.Image,
    model: Any,
    processor: Any,
    device_name: str,
    process_vision_info: Any,
) -> Dict[str, str]:
    raw_text = _infer_text(
        image=image,
        system_prompt="You are a helpful assistant. Output valid JSON only.",
        user_prompt=SCENE_SUMMARY_PROMPT,
        model=model,
        processor=processor,
        device_name=device_name,
        process_vision_info=process_vision_info,
        max_new_tokens=256,
    )
    payload = extract_json_payload(raw_text, expect_array=False)
    return {
        "scene_summary": _normalize_text(payload.get("scene_summary")) or "알 수 없음",
        "location_context": _normalize_location_context(
            payload.get("location_context", "알 수 없음")
        ),
    }


def _get_object_detail(
    *,
    image: Image.Image,
    object_name: str,
    object_id: int,
    model: Any,
    processor: Any,
    device_name: str,
    process_vision_info: Any,
) -> Dict[str, Any] | None:
    raw_text = _infer_text(
        image=image,
        system_prompt=SYSTEM_PROMPT_DETAIL,
        user_prompt=USER_PROMPT_DETAIL.format(object_name=object_name),
        model=model,
        processor=processor,
        device_name=device_name,
        process_vision_info=process_vision_info,
        max_new_tokens=256,
    )
    payload = extract_json_payload(raw_text, expect_array=False)
    normalized_objects = normalize_structured_objects({"objects": [payload]})
    if not normalized_objects:
        return None
    detail = normalized_objects[0]
    detail["object_id"] = object_id
    if detail.get("name") != object_name and object_name:
        detail["name"] = object_name
    return detail


def _deduplicate_with_vlm(
    *,
    image: Image.Image,
    object_list: List[str],
    model: Any,
    processor: Any,
    device_name: str,
    process_vision_info: Any,
) -> List[str]:
    clean_list = deduplicate_object_names(object_list)
    if not clean_list:
        return []

    raw_text = _infer_text(
        image=image,
        system_prompt=SYSTEM_PROMPT_LIST,
        user_prompt=DEDUP_PROMPT_TEMPLATE.format(
            object_list=json.dumps(clean_list, ensure_ascii=False)
        ),
        model=model,
        processor=processor,
        device_name=device_name,
        process_vision_info=process_vision_info,
        max_new_tokens=256,
    )
    reviewed = deduplicate_object_names(extract_json_payload(raw_text, expect_array=True))
    input_set = set(clean_list)
    reviewed = [name for name in reviewed if name in input_set]
    if not reviewed or len(reviewed) < max(1, len(clean_list) // 2):
        return clean_list
    return reviewed


@lru_cache(maxsize=4)
def get_qwen_vlm_components(
    model_key: str,
    quantization: str = "4bit",
    device: str = DEFAULT_DEVICE,
    dtype_name: str = "float16",
) -> Tuple[str, QwenVlmSpec, Any, Any]:
    spec = get_qwen_vlm_spec(model_key)
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError(
            "Installed transformers version does not support Qwen2.5-VL."
        ) from exc

    torch_dtype = _resolve_dtype(dtype_name)
    quantization_config = _build_quantization_config(quantization, torch_dtype)
    kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
    }
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = torch_dtype
        if device == "cuda":
            kwargs["device_map"] = "auto"

    started_at = time.perf_counter()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(spec.model_id, **kwargs)
    processor = AutoProcessor.from_pretrained(
        spec.model_id,
        trust_remote_code=True,
        max_pixels=spec.max_image_pixels,
    )
    load_time_sec = time.perf_counter() - started_at
    setattr(model, "_load_time_sec", load_time_sec)
    model.eval()
    return device, spec, model, processor


def generate_qwen_vlm_metadata(
    image: Image.Image,
    model_key: str = "qwen2.5-vl-3b",
    quantization: str = "4bit",
    device: Optional[str] = None,
    dtype_name: str = "float16",
    max_new_tokens: int = 512,
) -> Dict[str, Any]:
    process_vision_info = _require_qwen_dependencies()
    resolved_device = device or DEFAULT_DEVICE
    device_name, spec, model, processor = get_qwen_vlm_components(
        model_key=model_key,
        quantization=quantization,
        device=resolved_device,
        dtype_name=dtype_name,
    )
    original_image = image.convert("RGB")
    original_width, original_height = original_image.size
    max_image_pixels = _resolve_max_image_pixels(spec)
    resolved_image = _downscale_image_if_needed(
        original_image,
        max_image_pixels=max_image_pixels,
    )
    processed_width, processed_height = resolved_image.size

    if torch.cuda.is_available() and device_name.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    started_at = time.perf_counter()
    pipeline_profile = _new_pipeline_profile()
    object_list = _run_profiled_stage(
        pipeline_profile,
        "object_list",
        lambda: _get_object_list(
            image=resolved_image,
            model=model,
            processor=processor,
            device_name=device_name,
            process_vision_info=process_vision_info,
        ),
    )
    scene_info = _run_profiled_stage(
        pipeline_profile,
        "scene_summary",
        lambda: _get_scene_summary(
            image=resolved_image,
            model=model,
            processor=processor,
            device_name=device_name,
            process_vision_info=process_vision_info,
        ),
    )

    objects: List[Dict[str, Any]] = []
    for index, object_name in enumerate(object_list, start=1):
        detail = _run_profiled_stage(
            pipeline_profile,
            "object_detail",
            lambda object_name=object_name, index=index: _get_object_detail(
                image=resolved_image,
                object_name=object_name,
                object_id=index,
                model=model,
                processor=processor,
                device_name=device_name,
                process_vision_info=process_vision_info,
            ),
        )
        if detail is not None:
            objects.append(detail)

    initial_detail_count = len(objects)
    nearby_candidates = _run_profiled_stage(
        pipeline_profile,
        "nearby_candidate_collection",
        lambda: collect_nearby_candidate_names(objects),
    )
    all_candidates = [item["name"] for item in objects] + nearby_candidates
    deduped_list = _run_profiled_stage(
        pipeline_profile,
        "deduplicate",
        lambda: _deduplicate_with_vlm(
            image=resolved_image,
            object_list=all_candidates,
            model=model,
            processor=processor,
            device_name=device_name,
            process_vision_info=process_vision_info,
        ),
    )

    existing_names = {item["name"] for item in objects}
    for object_name in deduped_list:
        if object_name in existing_names:
            continue
        detail = _run_profiled_stage(
            pipeline_profile,
            "object_detail",
            lambda object_name=object_name: _get_object_detail(
                image=resolved_image,
                object_name=object_name,
                object_id=len(objects) + 1,
                model=model,
                processor=processor,
                device_name=device_name,
                process_vision_info=process_vision_info,
            ),
        )
        if detail is not None:
            objects.append(detail)
            existing_names.add(object_name)

    deduped_set = set(deduped_list)
    objects = [item for item in objects if item["name"] in deduped_set]

    missing_objects = _run_profiled_stage(
        pipeline_profile,
        "missing_object_check",
        lambda: _check_missing_objects_with_vlm(
            image=resolved_image,
            current_names=[item["name"] for item in objects],
            model=model,
            processor=processor,
            device_name=device_name,
            process_vision_info=process_vision_info,
        ),
    )
    for object_name in missing_objects:
        detail = _run_profiled_stage(
            pipeline_profile,
            "object_detail",
            lambda object_name=object_name: _get_object_detail(
                image=resolved_image,
                object_name=object_name,
                object_id=len(objects) + 1,
                model=model,
                processor=processor,
                device_name=device_name,
                process_vision_info=process_vision_info,
            ),
        )
        if detail is not None:
            objects.append(detail)

    for index, item in enumerate(objects, start=1):
        item["object_id"] = index

    elapsed_sec = time.perf_counter() - started_at
    scene_summary = scene_info.get("scene_summary") or None
    location_context = scene_info.get("location_context") or "알 수 없음"
    sharpness_score = _calculate_sharpness(resolved_image)
    vram_info = _get_vram_usage()
    pipeline_diagnostics = _build_pipeline_diagnostics(pipeline_profile)
    metadata = {
        "caption": scene_summary,
        "sceneSummary": scene_summary,
        "detectedObjects": deduplicate_object_names([item["name"] for item in objects]),
        "tags": [],
        "ocrText": None,
        "positionHint": _choose_position_hint(objects, location_context),
        "location": None,
    }
    metadata["tags"] = build_metadata_tags(metadata, objects)
    pipeline_output = {
        "capture_id": f"qwen-{int(started_at * 1000)}",
        "timestamp": datetime.now().isoformat(),
        "image_path": None,
        "sharpness_score": sharpness_score,
        "inference_time": round(elapsed_sec, 3),
        "scene_summary": scene_summary,
        "location_context": location_context,
        "objects": objects,
        "pipeline_meta": {
            "vlm_model": spec.model_id,
            "object_count": len(objects),
            "vram_allocated_gb": vram_info.get("allocated_gb"),
            "vram_peak_gb": vram_info.get("max_allocated_gb"),
            "vram_total_gb": vram_info.get("total_gb"),
            "stage_timings_sec": pipeline_diagnostics["stage_timings_sec"],
            "stage_call_counts": pipeline_diagnostics["stage_call_counts"],
            "stage_total_sec": pipeline_diagnostics["stage_total_sec"],
            "slowest_stage": pipeline_diagnostics["slowest_stage"],
            "candidate_counts": {
                "initial_objects": len(object_list),
                "initial_details": initial_detail_count,
                "nearby_candidates": len(nearby_candidates),
                "deduped_objects": len(deduped_list),
                "missing_objects": len(missing_objects),
                "final_objects": len(objects),
            },
            "image_pixels": {
                "original": original_width * original_height,
                "processed": processed_width * processed_height,
                "max_allowed": max_image_pixels,
                "downscaled": (original_width, original_height)
                != (processed_width, processed_height),
            },
        },
    }

    peak_memory_mb = 0.0
    if torch.cuda.is_available() and device_name.startswith("cuda"):
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return {
        "metadata": metadata,
        "elapsed_sec": round(elapsed_sec, 4),
        "peak_memory_mb": round(peak_memory_mb, 2),
        "load_time_sec": round(getattr(model, "_load_time_sec", 0.0), 4),
        "model_id": spec.model_id,
        "model_key": spec.key,
        "device": device_name,
        "quantization": quantization,
        "prompt": USER_PROMPT_LIST,
        "system_prompt": SYSTEM_PROMPT_LIST,
        "raw_output_text": None,
        "objects": objects,
        "scene_info": scene_info,
        "pipeline_output": pipeline_output,
        "pipeline_mode": "multi_stage",
    }
