import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from PIL import Image

from src.models.qwen_vlm import generate_qwen_vlm_metadata, get_qwen_vlm_spec


def _select_images(sample_dir: Path, limit: int) -> list[Path]:
    key_images = sorted(sample_dir.glob("key_*.*"))[: max(1, limit // 2)]
    wallet_images = sorted(sample_dir.glob("wallet_*.*"))[: max(1, limit - len(key_images))]
    selected = key_images + wallet_images
    if len(selected) < limit:
        remaining = sorted(sample_dir.glob("*.*"))
        seen = {path.name for path in selected}
        for path in remaining:
            if path.name in seen:
                continue
            selected.append(path)
            seen.add(path.name)
            if len(selected) >= limit:
                break
    return selected[:limit]


def _run_single_inference(image_path: Path, model_key: str) -> dict[str, Any]:
    with Image.open(image_path) as image:
        result = generate_qwen_vlm_metadata(
            image=image.convert("RGB"),
            model_key=model_key,
            quantization="4bit",
            dtype_name="float16",
        )

    metadata = result["metadata"]
    pipeline_meta = result.get("pipeline_output", {}).get("pipeline_meta", {})
    return {
        "status": "success",
        "modelKey": model_key,
        "image": image_path.name,
        "caption": metadata.get("caption"),
        "sceneSummary": metadata.get("sceneSummary"),
        "detectedObjects": metadata.get("detectedObjects", []),
        "tags": metadata.get("tags", []),
        "positionHint": metadata.get("positionHint"),
        "runtime": {
            "latencySec": result.get("elapsed_sec"),
            "peakMemoryMb": result.get("peak_memory_mb"),
            "loadTimeSec": result.get("load_time_sec"),
        },
        "pipelineDiagnostics": {
            "stageTimingsSec": pipeline_meta.get("stage_timings_sec", {}),
            "stageCallCounts": pipeline_meta.get("stage_call_counts", {}),
            "stageTotalSec": pipeline_meta.get("stage_total_sec"),
            "slowestStage": pipeline_meta.get("slowest_stage"),
            "candidateCounts": pipeline_meta.get("candidate_counts", {}),
            "imagePixels": pipeline_meta.get("image_pixels", {}),
        },
    }


def _run_single_case(image_path: Path, model_key: str) -> dict[str, Any]:
    try:
        return _run_single_inference(image_path, model_key)
    except Exception as exc:
        return {
            "status": "error",
            "modelKey": model_key,
            "image": image_path.name,
            "error": str(exc),
        }


def _build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [item for item in results if item["status"] == "success"]
    stage_timings: dict[str, list[float]] = {}
    slowest_stage_counts: dict[str, int] = {}
    for item in successes:
        diagnostics = item.get("pipelineDiagnostics", {})
        for stage_name, elapsed_sec in diagnostics.get("stageTimingsSec", {}).items():
            if isinstance(elapsed_sec, (int, float)):
                stage_timings.setdefault(stage_name, []).append(float(elapsed_sec))

        slowest_stage = diagnostics.get("slowestStage")
        if isinstance(slowest_stage, str) and slowest_stage:
            slowest_stage_counts[slowest_stage] = (
                slowest_stage_counts.get(slowest_stage, 0) + 1
            )

    return {
        "total": len(results),
        "successCount": len(successes),
        "avgLatencySec": round(mean(item["runtime"]["latencySec"] for item in successes), 4)
        if successes
        else None,
        "avgObjectCount": round(mean(len(item["detectedObjects"]) for item in successes), 2)
        if successes
        else None,
        "avgTagCount": round(mean(len(item["tags"]) for item in successes), 2)
        if successes
        else None,
        "avgCaptionLength": round(mean(len(item["caption"] or "") for item in successes), 2)
        if successes
        else None,
        "avgStageTimingsSec": {
            stage_name: round(mean(values), 4)
            for stage_name, values in sorted(stage_timings.items())
        },
        "slowestStageCounts": slowest_stage_counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run side-by-side batch comparison for Qwen VLM models."
    )
    parser.add_argument(
        "--sample-dir",
        default="/app/sample_data",
        help="Directory containing sample images",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=8,
        help="Number of sample images to compare",
    )
    parser.add_argument(
        "--output",
        default="/app/output/qwen-benchmark-report.json",
        help="Path to write the comparison report",
    )
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        default=None,
        help="Model key to include. Repeat for multiple models.",
    )
    args = parser.parse_args()

    sample_dir = Path(args.sample_dir)
    if not sample_dir.exists():
        raise FileNotFoundError(f"Sample directory not found: {sample_dir}")

    models = args.models or ["qwen2.5-vl-3b", "qwen2.5-vl-7b"]
    for model_key in models:
        get_qwen_vlm_spec(model_key)

    selected_images = _select_images(sample_dir, args.limit)
    if not selected_images:
        raise FileNotFoundError(f"No sample images found in: {sample_dir}")

    cases: list[dict[str, Any]] = []
    summary_by_model: dict[str, dict[str, Any]] = {}
    for model_key in models:
        model_results = [_run_single_case(image_path, model_key) for image_path in selected_images]
        summary_by_model[model_key] = _build_summary(model_results)
        cases.extend(model_results)

    report = {
        "sampleDir": str(sample_dir),
        "images": [path.name for path in selected_images],
        "models": models,
        "summaryByModel": summary_by_model,
        "results": cases,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
