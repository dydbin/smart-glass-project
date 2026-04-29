from dataclasses import dataclass

from src.models.captioning import get_caption_model_spec
from src.models.qwen_vlm import get_qwen_vlm_spec


@dataclass(frozen=True)
class InferenceModelCapabilities:
    caption: bool
    position_hint: bool
    scene_summary: bool
    detected_objects: bool
    tags: bool
    ocr_text: bool
    location: bool
    pipeline_output: bool

    def to_contract_payload(self) -> dict[str, bool]:
        return {
            "caption": self.caption,
            "positionHint": self.position_hint,
            "sceneSummary": self.scene_summary,
            "detectedObjects": self.detected_objects,
            "tags": self.tags,
            "ocrText": self.ocr_text,
            "location": self.location,
            "pipelineOutput": self.pipeline_output,
        }


@dataclass(frozen=True)
class InferenceModelDescriptor:
    key: str
    model_id: str
    family: str
    mode: str
    recommended_quantization: str
    notes: str
    capabilities: InferenceModelCapabilities


CAPTION_CAPABILITIES = InferenceModelCapabilities(
    caption=True,
    position_hint=True,
    scene_summary=False,
    detected_objects=False,
    tags=False,
    ocr_text=False,
    location=False,
    pipeline_output=False,
)

QWEN_VLM_CAPABILITIES = InferenceModelCapabilities(
    caption=True,
    position_hint=True,
    scene_summary=True,
    detected_objects=True,
    tags=True,
    ocr_text=False,
    location=False,
    pipeline_output=True,
)


def resolve_inference_model(model_key: str) -> InferenceModelDescriptor:
    try:
        spec = get_caption_model_spec(model_key)
        return InferenceModelDescriptor(
            key=spec.key,
            model_id=spec.model_id,
            family=spec.family,
            mode="caption",
            recommended_quantization=spec.recommended_quantization,
            notes=spec.notes,
            capabilities=CAPTION_CAPABILITIES,
        )
    except Exception:
        pass

    spec = get_qwen_vlm_spec(model_key)
    return InferenceModelDescriptor(
        key=spec.key,
        model_id=spec.model_id,
        family=spec.family,
        mode="vlm",
        recommended_quantization=spec.recommended_quantization,
        notes=spec.notes,
        capabilities=QWEN_VLM_CAPABILITIES,
    )
