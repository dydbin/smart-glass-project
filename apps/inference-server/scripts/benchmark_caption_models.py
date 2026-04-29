import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
APP_ROOT = SCRIPT_DIR.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    default_device = "cuda"
    parser = argparse.ArgumentParser(
        description="Benchmark image captioning models and store per-image / summary results."
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Directory containing evaluation images.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["blip-base", "git-base", "vit-gpt2"],
        help="Model keys to benchmark.",
    )
    parser.add_argument(
        "--quantizations",
        nargs="+",
        default=["none"],
        choices=["none", "8bit", "4bit"],
        help="Quantization modes to test.",
    )
    parser.add_argument(
        "--references",
        help="Optional CSV/JSONL file with image_path and reference_caption columns.",
    )
    parser.add_argument(
        "--output-dir",
        default="apps/inference-server/benchmark_results",
        help="Directory to write benchmark artifacts.",
    )
    parser.add_argument(
        "--device",
        default=default_device,
        help="Target device. Example: cuda, cpu, cuda:0",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Torch dtype used for non-quantized model loading.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        help="Cap the number of images for quick smoke tests.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Warm up count per model/quantization before measurement.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Maximum generated tokens.",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=3,
        help="Beam size for generation.",
    )
    parser.add_argument(
        "--latency-budget-sec",
        type=float,
        default=10.0,
        help="Latency budget used when choosing serving profile defaults.",
    )
    return parser.parse_args()


def collect_images(dataset_dir: str, max_images: Optional[int]) -> List[str]:
    paths: List[str] = []
    for root, _, files in os.walk(dataset_dir):
        for filename in sorted(files):
            extension = Path(filename).suffix.lower()
            if extension in IMAGE_EXTENSIONS:
                paths.append(str(Path(root) / filename))
    paths.sort()
    if max_images:
        return paths[:max_images]
    return paths


def load_references(reference_path: Optional[str]) -> Dict[str, str]:
    if not reference_path:
        return {}

    path = Path(reference_path)
    if not path.exists():
        raise FileNotFoundError(f"Reference file not found: {reference_path}")

    references: Dict[str, str] = {}
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                image_path = row.get("image_path", "").strip()
                reference_caption = row.get("reference_caption", "").strip()
                if image_path and reference_caption:
                    references[image_path] = reference_caption
    elif path.suffix.lower() in {".jsonl", ".json"}:
        with path.open("r", encoding="utf-8") as handle:
            if path.suffix.lower() == ".json":
                rows = json.load(handle)
            else:
                rows = [json.loads(line) for line in handle if line.strip()]
            for row in rows:
                image_path = str(row.get("image_path", "")).strip()
                reference_caption = str(row.get("reference_caption", "")).strip()
                if image_path and reference_caption:
                    references[image_path] = reference_caption
    else:
        raise ValueError("Supported reference formats: CSV, JSON, JSONL")

    return references


def tokenize(text: str) -> List[str]:
    return [token for token in text.lower().strip().split() if token]


def lexical_f1(prediction: str, reference: str) -> float:
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counts: Dict[str, int] = {}
    ref_counts: Dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1

    overlap = 0
    for token, pred_count in pred_counts.items():
        overlap += min(pred_count, ref_counts.get(token, 0))

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * p
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return values[lower]
    weight = rank - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def ensure_output_dir(path: str) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    from src.models.captioning import (
        generate_caption,
        get_caption_model_spec,
        open_image,
    )
    from src.models.serving_profile import build_caption_serving_profile

    image_paths = collect_images(args.dataset_dir, args.max_images)
    if not image_paths:
        raise ValueError(f"No images found under {args.dataset_dir}")

    references = load_references(args.references)
    output_dir = ensure_output_dir(args.output_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    per_image_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    for model_key in args.models:
        spec = get_caption_model_spec(model_key)
        for quantization in args.quantizations:
            print(f"\n=== Benchmark: {model_key} / {quantization} ===")
            print(f"Model ID: {spec.model_id}")

            if args.warmup_runs > 0:
                warmup_image = open_image(image_paths[0])
                for warmup_idx in range(args.warmup_runs):
                    print(
                        f"Warmup {warmup_idx + 1}/{args.warmup_runs} for {model_key} ({quantization})"
                    )
                    _ = generate_caption(
                        image=warmup_image,
                        model_key=model_key,
                        quantization=quantization,
                        device=args.device,
                        dtype_name=args.dtype,
                        max_new_tokens=args.max_new_tokens,
                        num_beams=args.num_beams,
                    )

            run_rows: List[Dict[str, object]] = []
            load_time_sec: Optional[float] = None
            started_run = time.perf_counter()

            for image_path in image_paths:
                try:
                    image = open_image(image_path)
                    result = generate_caption(
                        image=image,
                        model_key=model_key,
                        quantization=quantization,
                        device=args.device,
                        dtype_name=args.dtype,
                        max_new_tokens=args.max_new_tokens,
                        num_beams=args.num_beams,
                    )
                    load_time_sec = result["load_time_sec"]
                    reference = references.get(image_path, "")
                    row = {
                        "model_key": model_key,
                        "model_id": result["model_id"],
                        "quantization": quantization,
                        "device": args.device,
                        "dtype": args.dtype,
                        "image_path": image_path,
                        "caption": result["caption"],
                        "reference_caption": reference,
                        "lexical_f1": round(lexical_f1(result["caption"], reference), 4)
                        if reference
                        else "",
                        "latency_sec": round(result["elapsed_sec"], 4),
                        "peak_memory_mb": result["peak_memory_mb"],
                        "load_time_sec": result["load_time_sec"],
                        "status": "success",
                        "error": "",
                    }
                except Exception as exc:
                    row = {
                        "model_key": model_key,
                        "model_id": spec.model_id,
                        "quantization": quantization,
                        "device": args.device,
                        "dtype": args.dtype,
                        "image_path": image_path,
                        "caption": "",
                        "reference_caption": references.get(image_path, ""),
                        "lexical_f1": "",
                        "latency_sec": "",
                        "peak_memory_mb": "",
                        "load_time_sec": load_time_sec if load_time_sec is not None else "",
                        "status": "error",
                        "error": str(exc),
                    }

                per_image_rows.append(row)
                run_rows.append(row)
                print(
                    f"[{row['status']}] {Path(image_path).name} -> {str(row['caption'])[:80]}"
                )

            run_elapsed = time.perf_counter() - started_run
            success_rows = [row for row in run_rows if row["status"] == "success"]
            latency_values = [float(row["latency_sec"]) for row in success_rows]
            memory_values = [float(row["peak_memory_mb"]) for row in success_rows]
            lexical_values = [
                float(row["lexical_f1"])
                for row in success_rows
                if row["lexical_f1"] != ""
            ]

            summary = {
                "model_key": model_key,
                "model_id": spec.model_id,
                "quantization": quantization,
                "device": args.device,
                "dtype": args.dtype,
                "images_total": len(run_rows),
                "images_success": len(success_rows),
                "images_error": len(run_rows) - len(success_rows),
                "load_time_sec": round(load_time_sec or 0.0, 4),
                "latency_avg_sec": round(statistics.mean(latency_values), 4)
                if latency_values
                else "",
                "latency_p50_sec": round(percentile(sorted(latency_values), 0.50), 4)
                if latency_values
                else "",
                "latency_p95_sec": round(percentile(sorted(latency_values), 0.95), 4)
                if latency_values
                else "",
                "peak_memory_avg_mb": round(statistics.mean(memory_values), 2)
                if memory_values
                else "",
                "peak_memory_max_mb": round(max(memory_values), 2)
                if memory_values
                else "",
                "lexical_f1_avg": round(statistics.mean(lexical_values), 4)
                if lexical_values
                else "",
                "run_elapsed_sec": round(run_elapsed, 4),
            }
            summary_rows.append(summary)

            print("Summary:", json.dumps(summary, ensure_ascii=False))

    per_image_csv = output_dir / f"caption_benchmark_per_image_{timestamp}.csv"
    summary_csv = output_dir / f"caption_benchmark_summary_{timestamp}.csv"
    summary_json = output_dir / f"caption_benchmark_summary_{timestamp}.json"

    write_csv(per_image_csv, per_image_rows)
    write_csv(summary_csv, summary_rows)
    summary_payload = {
        "generated_at_utc": timestamp,
        "dataset_dir": args.dataset_dir,
        "device": args.device,
        "dtype": args.dtype,
        "models": args.models,
        "quantizations": args.quantizations,
        "results": summary_rows,
    }
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(
            summary_payload,
            handle,
            ensure_ascii=False,
            indent=2,
        )

    serving_profile_json = output_dir / f"caption_serving_profile_{timestamp}.json"
    serving_profile_latest = output_dir / "caption_serving_profile.json"
    serving_profile_payload = build_caption_serving_profile(
        summary_payload,
        latency_budget_sec=args.latency_budget_sec,
        source_summary_path=str(summary_json),
    )
    for path in (serving_profile_json, serving_profile_latest):
        with path.open("w", encoding="utf-8") as handle:
            json.dump(serving_profile_payload, handle, ensure_ascii=False, indent=2)

    print(f"\nPer-image results: {per_image_csv}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Summary JSON: {summary_json}")
    print(f"Serving profile JSON: {serving_profile_json}")
    print(f"Serving profile latest: {serving_profile_latest}")


if __name__ == "__main__":
    main()
