import unittest

from src.contracts.qwen_parser import (
    build_metadata_tags,
    collect_nearby_candidate_names,
    deduplicate_object_names,
    merge_detected_object_candidates,
    extract_json_payload,
    normalize_qwen_metadata,
    object_names_from_structured_payload,
)


class QwenParserTestCase(unittest.TestCase):
    def test_extract_json_payload_reads_code_fenced_object(self) -> None:
        payload = extract_json_payload(
            """```json
            {"caption":"지갑이 키보드 옆 책상 위에 있다","tags":["지갑","키보드"]}
            ```"""
        )

        self.assertEqual(payload["caption"], "지갑이 키보드 옆 책상 위에 있다")
        self.assertEqual(payload["tags"], ["지갑", "키보드"])

    def test_normalize_qwen_metadata_backfills_position_hint(self) -> None:
        metadata = normalize_qwen_metadata(
            {
                "caption": "wallet is on the desk next to the keyboard",
                "scene_summary": "책상 위 장면",
                "objects": [{"name": "지갑"}, {"name": "키보드"}],
            }
        )

        self.assertEqual(metadata["caption"], "wallet is on the desk next to the keyboard")
        self.assertEqual(metadata["sceneSummary"], "책상 위 장면")
        self.assertEqual(metadata["positionHint"], "keyboard 옆")

    def test_object_names_from_structured_payload_preserves_unique_names(self) -> None:
        names = object_names_from_structured_payload(
            {
                "objects": [
                    {"name": "지갑"},
                    {"name": "키보드"},
                    {"name": "지갑"},
                    {"surface": "책상 위"},
                ]
            }
        )

        self.assertEqual(names, ["지갑", "키보드"])

    def test_deduplicate_object_names_prefers_more_specific_alias(self) -> None:
        names = deduplicate_object_names(
            ["노트북", "맥북", "스마트폰", "아이폰", "airpods", "AirPods 케이스"]
        )

        self.assertEqual(names, ["맥북", "아이폰", "AirPods 케이스"])

    def test_collect_nearby_candidate_names_filters_existing_and_noise(self) -> None:
        candidates = collect_nearby_candidate_names(
            [
                {"name": "맥북", "nearby_objects": ["아이폰", "airpods", "충전기"]},
                {"name": "아이폰", "nearby_objects": ["맥북", "애플펜슬"]},
            ],
            ["맥북", "아이폰"],
        )

        self.assertEqual(candidates, ["충전기", "애플펜슬"])

    def test_merge_detected_object_candidates_combines_structured_and_nearby(self) -> None:
        names = merge_detected_object_candidates(
            detected_objects=["노트북"],
            structured_objects=[
                {"name": "맥북", "nearby_objects": ["아이폰"]},
                {"name": "에어팟 케이스", "nearby_objects": ["맥북"]},
            ],
        )

        self.assertEqual(names, ["맥북", "에어팟 케이스", "아이폰"])

    def test_build_metadata_tags_aggregates_brand_and_position(self) -> None:
        tags = build_metadata_tags(
            {
                "positionHint": "책상 위",
                "tags": ["업무 공간"],
                "detectedObjects": ["맥북"],
            },
            [
                {
                    "name": "맥북",
                    "nearby_objects": ["아이폰"],
                    "visual_features": {"brand": "Apple"},
                }
            ],
        )

        self.assertEqual(tags, ["책상 위", "업무 공간", "맥북", "아이폰", "Apple"])

    def test_normalize_qwen_metadata_ignores_dict_like_detected_objects(self) -> None:
        metadata = normalize_qwen_metadata(
            {
                "detectedObjects": [{"name": "고양이"}, "아이폰", "작업 공간"],
                "tags": ["아이폰"],
            }
        )

        self.assertEqual(metadata["detectedObjects"], ["아이폰"])


if __name__ == "__main__":
    unittest.main()
