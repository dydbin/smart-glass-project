import unittest

from scripts.benchmark_qwen_models import _build_summary


class QwenBenchmarkSummaryTestCase(unittest.TestCase):
    def test_build_summary_aggregates_pipeline_diagnostics(self) -> None:
        summary = _build_summary(
            [
                {
                    "status": "success",
                    "runtime": {"latencySec": 10.0},
                    "detectedObjects": ["맥북", "아이폰"],
                    "tags": ["전자기기"],
                    "caption": "책상 위 전자기기 장면",
                    "pipelineDiagnostics": {
                        "stageTimingsSec": {
                            "object_list": 1.0,
                            "object_detail": 6.0,
                        },
                        "slowestStage": "object_detail",
                    },
                },
                {
                    "status": "success",
                    "runtime": {"latencySec": 14.0},
                    "detectedObjects": ["지갑"],
                    "tags": ["지갑", "책상"],
                    "caption": "책상 위 지갑",
                    "pipelineDiagnostics": {
                        "stageTimingsSec": {
                            "object_list": 2.0,
                            "object_detail": 8.0,
                        },
                        "slowestStage": "object_detail",
                    },
                },
                {
                    "status": "error",
                    "error": "boom",
                },
            ]
        )

        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["successCount"], 2)
        self.assertEqual(summary["avgLatencySec"], 12.0)
        self.assertEqual(
            summary["avgStageTimingsSec"],
            {
                "object_detail": 7.0,
                "object_list": 1.5,
            },
        )
        self.assertEqual(summary["slowestStageCounts"], {"object_detail": 2})


if __name__ == "__main__":
    unittest.main()
