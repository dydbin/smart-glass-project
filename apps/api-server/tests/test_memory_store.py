from __future__ import annotations

import unittest

from src.database.memory_store import memory_record_from_vlm_result


class MemoryStoreAdapterTests(unittest.TestCase):
    def test_memory_record_from_vlm_result_normalizes_success_payload(self) -> None:
        record = memory_record_from_vlm_result(
            {
                "status": "success",
                "memoryId": "mem-101",
                "userId": "user-101",
                "capturedAt": "2026-04-17T10:23:45+09:00",
                "sourceImage": {
                    "imageKey": "captures/user-101/2026/04/17/mem-101-photo.jpg",
                    "imageUrl": "https://example.com/photo.jpg",
                },
                "metadata": {
                    "caption": "wallet on the desk",
                    "sceneSummary": "desk scene",
                    "detectedObjects": ["wallet", "desk"],
                    "tags": ["workspace", "desk"],
                    "ocrText": "meeting notes",
                    "positionHint": "on the desk",
                    "location": {"name": "office"},
                },
                "pipelineOutput": {
                    "scene_summary": "desk scene",
                    "location_context": "office desk",
                },
                "providerMetadata": {
                    "capabilities": {
                        "detectedObjects": True,
                        "tags": True,
                        "positionHint": True,
                        "sceneSummary": True,
                        "ocrText": True,
                        "location": True,
                    }
                },
            }
        )

        self.assertEqual(record.memory_id, "mem-101")
        self.assertEqual(record.user_id, "user-101")
        self.assertEqual(record.image_key, "captures/user-101/2026/04/17/mem-101-photo.jpg")
        self.assertEqual(record.captured_at, "2026-04-17T01:23:45Z")
        self.assertEqual(record.caption, "wallet on the desk")
        self.assertEqual(record.position_hint, "on the desk")
        self.assertEqual(record.location.name, "office")
        self.assertEqual(record.tags, ["workspace", "desk"])

    def test_memory_record_from_vlm_result_uses_pipeline_objects_when_capabilities_absent(self) -> None:
        record = memory_record_from_vlm_result(
            {
                "status": "success",
                "memoryId": "mem-202",
                "userId": "user-202",
                "sourceImage": {"imageKey": "captures/user-202/mem-202-photo.jpg"},
                "metadata": {
                    "caption": "keys beside a mug",
                    "detectedObjects": [],
                    "tags": [],
                },
                "pipelineOutput": {
                    "scene_summary": "keys near a mug",
                    "objects": [
                        {
                            "name": "keys",
                            "nearby_objects": ["mug"],
                            "visual_features": {"brand": "Starbucks"},
                        }
                    ],
                },
            }
        )

        self.assertEqual(record.detected_objects, ["keys"])
        self.assertEqual(record.tags, ["keys", "mug", "Starbucks"])
        self.assertEqual(record.position_hint, "mug 옆")

    def test_memory_record_from_vlm_result_rejects_failed_payload(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Only successful VLM inference results can be stored",
        ):
            memory_record_from_vlm_result({"status": "error"})

    def test_memory_record_from_vlm_result_rejects_invalid_timestamp(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "capturedAt must be a valid ISO 8601 timestamp",
        ):
            memory_record_from_vlm_result(
                {
                    "status": "success",
                    "memoryId": "mem-303",
                    "userId": "user-303",
                    "capturedAt": "not-a-timestamp",
                    "sourceImage": {"imageKey": "captures/user-303/photo.jpg"},
                    "metadata": {"caption": "wallet on the desk"},
                    "pipelineOutput": {"objects": []},
                }
            )


if __name__ == "__main__":
    unittest.main()
