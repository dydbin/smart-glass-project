from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, validator

try:
    from pydantic import ConfigDict
except ImportError:  # pragma: no cover - pydantic v1 fallback
    ConfigDict = None


class ApiSchema(BaseModel):
    if ConfigDict is not None:
        model_config = ConfigDict(
            extra="ignore",
            str_strip_whitespace=True,
            populate_by_name=True,
        )
    else:
        class Config:
            extra = "ignore"
            anystr_strip_whitespace = True
            allow_population_by_field_name = True


class CaptureSourceImagePayload(ApiSchema):
    imageKey: str | None = None
    imageUrl: str | None = None
    contentType: str | None = None


class GenerationConfigPayload(ApiSchema):
    modelKey: str | None = None
    quantization: Literal["none", "8bit", "4bit"] | None = None
    dtype: Literal["float16", "bfloat16", "float32"] | None = None
    prompt: str | None = None
    maxNewTokens: int | None = Field(default=None, ge=1)
    numBeams: int | None = Field(default=None, ge=1)


class CaptureUploadRequest(ApiSchema):

    captureId: str | None = None
    requestId: str | None = None
    memoryId: str | None = None
    userId: str = Field(min_length=1)
    taskType: Literal["caption", "metadata"] = "metadata"
    capturedAt: str | None = None
    fileName: str | None = None
    imageKey: str | None = None
    imageUrl: str | None = None
    contentType: str | None = None
    sourceImage: CaptureSourceImagePayload | None = None
    generation: GenerationConfigPayload | None = None

    @validator("userId")
    def validate_non_blank_user_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be blank")
        return normalized


class VlmSourceImagePayload(ApiSchema):
    imageKey: str
    imageUrl: str | None = None
    contentType: str | None = None


class VlmGenerationConfigPayload(ApiSchema):
    modelKey: str | None = None
    quantization: Literal["none", "8bit", "4bit"] | None = None
    dtype: Literal["float16", "bfloat16", "float32"] | None = None
    prompt: str | None = None
    maxNewTokens: int | None = Field(default=None, ge=1)
    numBeams: int | None = Field(default=None, ge=1)


class VlmInferenceRequestPayload(ApiSchema):
    requestId: str
    taskType: Literal["caption", "metadata"]
    memoryId: str | None = None
    userId: str
    capturedAt: str | None = None
    sourceImage: VlmSourceImagePayload
    generation: VlmGenerationConfigPayload | None = None


class CaptureSourceImageSnapshot(ApiSchema):
    imageKey: str
    imageUrl: str | None = None
    contentType: str | None = None
    fileName: str | None = None


class CaptureDispatchPlan(ApiSchema):
    transport: Literal["celery-redis"] = "celery-redis"
    taskName: Literal["process_vision_inference"] = "process_vision_inference"
    status: Literal["prepared"] = "prepared"
    brokerUrl: str | None = None
    note: str


class CaptureUploadResponse(ApiSchema):
    status: Literal["accepted"] = "accepted"
    service: Literal["api-server"] = "api-server"
    captureId: str
    requestId: str
    memoryId: str
    userId: str
    taskType: Literal["caption", "metadata"]
    capturedAt: str
    sourceImage: CaptureSourceImageSnapshot
    inferenceRequest: VlmInferenceRequestPayload
    dispatch: CaptureDispatchPlan


class CaptureWorkerExecutionPayload(ApiSchema):
    taskId: str | None = None
    status: Literal["success", "error", "timeout"]
    result: dict[str, Any] | None = None
    error: str | None = None


class CaptureMemoryStoreExecutionPayload(ApiSchema):
    backend: str
    status: Literal["success", "error", "skipped"]
    storedCount: int | None = None
    totalUserMemories: dict[str, int] = Field(default_factory=dict)
    error: str | None = None


class CaptureProcessingResponse(ApiSchema):
    status: Literal["completed", "partial", "failed"]
    service: Literal["api-server"] = "api-server"
    capture: CaptureUploadResponse
    worker: CaptureWorkerExecutionPayload
    memoryStore: CaptureMemoryStoreExecutionPayload | None = None


class MediaAccessUrlRequest(ApiSchema):
    userId: str = Field(min_length=1)
    imageKey: str = Field(min_length=1)
    expiresInSec: int = Field(default=300, ge=30, le=3600)

    @validator("userId", "imageKey")
    def validate_non_blank_media_access_fields(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be blank")
        return normalized


class MediaAccessUrlResponse(ApiSchema):
    imageKey: str
    accessUrl: str
    expiresAt: str
    expiresInSec: int


class MediaBatchAccessUrlRequest(ApiSchema):
    userId: str = Field(min_length=1)
    imageKeys: list[str] = Field(min_items=1, max_items=100)
    expiresInSec: int = Field(default=300, ge=30, le=3600)

    @validator("userId")
    def validate_non_blank_media_batch_user_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be blank")
        return normalized

    @validator("imageKeys")
    def validate_image_keys(cls, image_keys: list[str]) -> list[str]:
        normalized_values = [
            " ".join(value.split())
            for value in image_keys
            if " ".join(value.split())
        ]
        if not normalized_values:
            raise ValueError("must include at least one non-blank imageKey")
        return normalized_values


class MediaBatchAccessUrlResponse(ApiSchema):
    totalItems: int
    items: list[MediaAccessUrlResponse]


class MediaGalleryRequest(ApiSchema):
    userId: str = Field(min_length=1)
    limit: int = Field(default=50, ge=1, le=200)

    @validator("userId")
    def validate_non_blank_media_gallery_user_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be blank")
        return normalized


class MediaGalleryItemPayload(ApiSchema):
    memoryId: str
    imageKey: str
    imageUrl: str | None = None
    capturedAt: str | None = None
    caption: str | None = None
    sceneSummary: str | None = None
    positionHint: str | None = None


class MediaGalleryResponse(ApiSchema):
    totalItems: int
    items: list[MediaGalleryItemPayload]


class MemoryLocationPayload(ApiSchema):
    name: str | None = None
    address: str | None = None
    latitude: float | None = None
    longitude: float | None = None


class MemorySearchRequest(ApiSchema):
    userId: str = Field(alias="user_id", min_length=1)
    query: str = Field(min_length=1)
    topK: int = Field(default=5, alias="top_k", ge=1, le=20)

    @validator("userId", "query")
    def validate_non_blank_search_fields(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be blank")
        return normalized


class MemoryChatRequest(MemorySearchRequest):
    conversationId: str | None = Field(default=None, alias="conversation_id")


class MemorySearchHitPayload(ApiSchema):
    memoryId: str
    score: float
    lexicalScore: float
    matchedTerms: list[str]
    imageKey: str | None = None
    imageUrl: str | None = None
    capturedAt: str | None = None
    caption: str | None = None
    sceneSummary: str | None = None
    positionHint: str | None = None
    location: MemoryLocationPayload
    detectedObjects: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class MemorySearchResponse(ApiSchema):
    query: str
    totalHits: int
    hits: list[MemorySearchHitPayload]


class MemoryChatResponse(ApiSchema):
    answer: str
    answerMode: str
    query: str
    totalHits: int
    hits: list[MemorySearchHitPayload]
    citedMemoryIds: list[str] = Field(default_factory=list)
    confidence: float | None = None
    reason: str | None = None


class HealthPayload(ApiSchema):
    status: Literal["ok", "error"]
    service: Literal["api-server"]
    checkType: Literal["liveness", "readiness"]
    timestamp: str
