import unittest
from unittest.mock import patch

from PIL import Image

from src.models.qwen_vlm import generate_qwen_vlm_metadata, get_qwen_vlm_spec


class _DummyModel:
    _load_time_sec = 3.2


class QwenPipelineTestCase(unittest.TestCase):
    def test_generate_qwen_vlm_metadata_uses_multi_stage_pipeline(self) -> None:
        image = Image.new("RGB", (64, 64), color=(255, 255, 255))
        spec = get_qwen_vlm_spec("qwen2.5-vl-7b")

        detail_map = {
            "맥북": {
                "object_id": 1,
                "name": "맥북",
                "position": {"depth_hint": "mid", "surface": "책상 위"},
                "nearby_objects": ["아이폰"],
                "visual_features": {
                    "color": "silver",
                    "material": "metal",
                    "brand": "Apple",
                    "shape": "rectangular",
                },
                "raw_description_ko": "책상 위 맥북",
                "confidence": 0.8,
            },
            "아이폰": {
                "object_id": 2,
                "name": "아이폰",
                "position": {"depth_hint": "near", "surface": "맥북 오른쪽"},
                "nearby_objects": ["맥북"],
                "visual_features": {
                    "color": "black",
                    "material": "glass",
                    "brand": "Apple",
                    "shape": "slim",
                },
                "raw_description_ko": "맥북 옆 아이폰",
                "confidence": 0.8,
            },
            "에어팟 케이스": {
                "object_id": 3,
                "name": "에어팟 케이스",
                "position": {"depth_hint": "near", "surface": "아이폰 옆"},
                "nearby_objects": ["아이폰"],
                "visual_features": {
                    "color": "white",
                    "material": "plastic",
                    "brand": "Apple",
                    "shape": "case",
                },
                "raw_description_ko": "아이폰 옆 에어팟 케이스",
                "confidence": 0.8,
            },
        }

        with patch("src.models.qwen_vlm._require_qwen_dependencies", return_value=object()):
            with patch(
                "src.models.qwen_vlm.get_qwen_vlm_components",
                return_value=("cpu", spec, _DummyModel(), object()),
            ):
                with patch(
                    "src.models.qwen_vlm._get_object_list",
                    return_value=["맥북", "아이폰"],
                ):
                    with patch(
                        "src.models.qwen_vlm._get_scene_summary",
                        return_value={
                            "scene_summary": "책상 위 전자기기 장면",
                            "location_context": "작업 공간",
                        },
                    ):
                        with patch(
                            "src.models.qwen_vlm._get_object_detail",
                            side_effect=lambda **kwargs: detail_map.get(kwargs["object_name"]),
                        ) as mocked_detail:
                            with patch(
                                "src.models.qwen_vlm._deduplicate_with_vlm",
                                return_value=["맥북", "아이폰"],
                            ):
                                with patch(
                                    "src.models.qwen_vlm._check_missing_objects_with_vlm",
                                    return_value=["에어팟 케이스"],
                                ):
                                    result = generate_qwen_vlm_metadata(
                                        image=image,
                                        model_key="qwen2.5-vl-7b",
                                    )

        self.assertEqual(result["pipeline_mode"], "multi_stage")
        self.assertEqual(result["scene_info"]["scene_summary"], "책상 위 전자기기 장면")
        self.assertEqual(
            result["metadata"]["detectedObjects"],
            ["맥북", "아이폰", "에어팟 케이스"],
        )
        self.assertEqual(result["metadata"]["caption"], "책상 위 전자기기 장면")
        self.assertEqual(result["metadata"]["positionHint"], "책상 위")
        self.assertIn("Apple", result["metadata"]["tags"])
        self.assertEqual(mocked_detail.call_count, 3)

        pipeline_meta = result["pipeline_output"]["pipeline_meta"]
        self.assertEqual(
            pipeline_meta["stage_call_counts"],
            {
                "object_list": 1,
                "scene_summary": 1,
                "object_detail": 3,
                "nearby_candidate_collection": 1,
                "deduplicate": 1,
                "missing_object_check": 1,
            },
        )
        self.assertEqual(
            set(pipeline_meta["stage_timings_sec"]),
            set(pipeline_meta["stage_call_counts"]),
        )
        self.assertIn(
            pipeline_meta["slowest_stage"],
            pipeline_meta["stage_timings_sec"],
        )
        self.assertGreaterEqual(pipeline_meta["stage_total_sec"], 0.0)
        self.assertEqual(
            pipeline_meta["candidate_counts"],
            {
                "initial_objects": 2,
                "initial_details": 2,
                "nearby_candidates": 0,
                "deduped_objects": 2,
                "missing_objects": 1,
                "final_objects": 3,
            },
        )
        self.assertEqual(
            pipeline_meta["image_pixels"],
            {
                "original": 4096,
                "processed": 4096,
                "max_allowed": spec.max_image_pixels,
                "downscaled": False,
            },
        )


if __name__ == "__main__":
    unittest.main()
