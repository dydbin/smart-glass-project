from typing import Any, Literal

from pydantic import BaseModel, Field


VisionTaskType = Literal["caption", "metadata"]


class VisionInferenceEnqueueRequest(BaseModel):
    image_key: str = Field(..., alias="imageKey")
    user_id: str = Field(..., alias="userId")
    memory_id: str | None = Field(default=None, alias="memoryId")
    captured_at: str | None = Field(default=None, alias="capturedAt")
    image_url: str | None = Field(default=None, alias="imageUrl")
    request_id: str | None = Field(default=None, alias="requestId")
    task_type: VisionTaskType = Field(default="metadata", alias="taskType")

    model_config = {"populate_by_name": True}


class VisionInferenceEnqueueResponse(BaseModel):
    task_id: str = Field(..., alias="taskId")
    state: str
    request_id: str = Field(..., alias="requestId")
    task_type: VisionTaskType = Field(..., alias="taskType")
    status_url: str = Field(..., alias="statusUrl")

    model_config = {"populate_by_name": True}


class VisionInferenceTaskStatusResponse(BaseModel):
    task_id: str = Field(..., alias="taskId")
    state: str
    ready: bool
    successful: bool
    result_status: Literal["success", "error"] | None = Field(
        default=None,
        alias="resultStatus",
    )
    request_id: str | None = Field(default=None, alias="requestId")
    result: dict[str, Any] | None = None
    error: str | None = None
    task_status: Literal["pending", "running", "completed", "failed"] = Field(
        ..., alias="taskStatus"
    )

    model_config = {"populate_by_name": True}
