export type VlmTaskStatus = "queued" | "processing" | "success" | "error";

export type VlmModelTask = "caption" | "metadata";

export interface VlmGenerationConfig {
  modelKey?: string;
  quantization?: "none" | "8bit" | "4bit";
  dtype?: "float16" | "bfloat16" | "float32";
  prompt?: string | null;
  maxNewTokens?: number;
  numBeams?: number;
}

export interface VlmSourceImageRef {
  imageKey?: string | null;
  imageUrl?: string | null;
  contentType?: string | null;
}

export interface VlmInferenceRequest {
  requestId: string;
  taskType: VlmModelTask;
  memoryId?: string | null;
  userId: string;
  capturedAt?: string | null;
  sourceImage: VlmSourceImageRef;
  generation?: VlmGenerationConfig;
}

export interface VlmLocationPayload {
  name?: string | null;
  address?: string | null;
  latitude?: number | null;
  longitude?: number | null;
}

export interface VlmMetadataPayload {
  caption: string | null;
  sceneSummary: string | null;
  detectedObjects: string[];
  tags: string[];
  ocrText: string | null;
  positionHint: string | null;
  location: VlmLocationPayload | null;
}

export interface VlmRuntimeMetrics {
  latencySec?: number | null;
  peakMemoryMb?: number | null;
  loadTimeSec?: number | null;
}

export interface VlmExecutionPolicyMetadata {
  settingsSource?: string | null;
  profilePath?: string | null;
  selectedModelKey?: string | null;
  softTimeLimitSec?: number | null;
  hardTimeLimitSec?: number | null;
  fallbackModelKey?: string | null;
  fallbackTriggered?: boolean | null;
}

export interface VlmProviderMetadata {
  capabilities?: VlmProviderCapabilities | null;
  executionPolicy?: VlmExecutionPolicyMetadata | null;
  modelKey?: string | null;
  modelId?: string | null;
  modelFamily?: string | null;
  quantization?: string | null;
  dtype?: string | null;
  provider?: string | null;
  raw?: Record<string, unknown> | null;
}

export interface VlmProviderCapabilities {
  caption?: boolean;
  positionHint?: boolean;
  sceneSummary?: boolean;
  detectedObjects?: boolean;
  tags?: boolean;
  ocrText?: boolean;
  location?: boolean;
  pipelineOutput?: boolean;
}

export interface VlmPipelineObjectPosition {
  depth_hint: string;
  surface: string;
}

export interface VlmPipelineObjectVisualFeatures {
  color: string;
  material: string;
  brand?: string | null;
  shape: string;
}

export interface VlmPipelineObject {
  object_id: number;
  name: string;
  confidence: number;
  position: VlmPipelineObjectPosition;
  visual_features: VlmPipelineObjectVisualFeatures;
  nearby_objects: string[];
  raw_description_ko?: string | null;
}

export interface VlmPipelineMeta {
  vlm_model?: string | null;
  object_count?: number | null;
  vram_allocated_gb?: number | null;
  vram_peak_gb?: number | null;
  vram_total_gb?: number | null;
}

export interface VlmPipelineOutput {
  capture_id: string;
  timestamp: string;
  image_path?: string | null;
  sharpness_score?: number | null;
  inference_time?: number | null;
  scene_summary?: string | null;
  location_context?: string | null;
  objects: VlmPipelineObject[];
  pipeline_meta?: VlmPipelineMeta | null;
}

export interface VlmInferenceSuccess {
  status: "success";
  requestId: string;
  taskType: VlmModelTask;
  memoryId?: string | null;
  userId: string;
  capturedAt: string | null;
  sourceImage: VlmSourceImageRef;
  metadata: VlmMetadataPayload;
  pipelineOutput?: VlmPipelineOutput | null;
  providerMetadata: VlmProviderMetadata;
  runtime: VlmRuntimeMetrics;
}

export interface VlmInferenceError {
  status: "error";
  requestId: string;
  taskType: VlmModelTask;
  memoryId?: string | null;
  userId: string;
  capturedAt: string | null;
  sourceImage: VlmSourceImageRef;
  errorCode: string;
  message: string;
  retryable?: boolean;
  providerMetadata: VlmProviderMetadata;
}

export type VlmInferenceResult = VlmInferenceSuccess | VlmInferenceError;
