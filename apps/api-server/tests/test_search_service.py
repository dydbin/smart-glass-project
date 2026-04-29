from __future__ import annotations

import unittest

from src.database.memory_store import MemoryLocation, MemoryRecord
from src.modules.search.service import MemoryQueryService


class FakeMemoryRepository:
    def __init__(self, records: list[MemoryRecord]) -> None:
        self.records = records
        self.last_user_id: str | None = None
        self.last_limit: int | None = None
        self.health_checked = False

    def list_by_user(self, user_id: str, *, limit: int | None = None) -> list[MemoryRecord]:
        self.last_user_id = user_id
        self.last_limit = limit
        return [record for record in self.records if record.user_id == user_id]

    def check_health(self) -> None:
        self.health_checked = True


class MemoryQueryServiceTests(unittest.TestCase):
    def test_search_ranks_matching_record(self) -> None:
        repository = FakeMemoryRepository(
            [
                MemoryRecord(
                    memory_id="mem-wallet-01",
                    user_id="user-1",
                    image_key="captures/wallet-01.jpg",
                    image_url=None,
                    captured_at="2026-04-17T09:00:00Z",
                    caption="wallet on the desk next to the keyboard",
                    scene_summary="workspace desk scene",
                    detected_objects=["wallet", "desk", "keyboard"],
                    tags=["office"],
                    ocr_text=None,
                    note=None,
                    position_hint="keyboard 옆",
                    location=MemoryLocation(name="workspace"),
                ),
                MemoryRecord(
                    memory_id="mem-umbrella-01",
                    user_id="user-1",
                    image_key="captures/umbrella-01.jpg",
                    image_url=None,
                    captured_at="2026-04-17T08:00:00Z",
                    caption="umbrella leaning against sofa",
                    scene_summary="living room scene",
                    detected_objects=["umbrella", "sofa"],
                    tags=["living-room"],
                    ocr_text=None,
                    note=None,
                    position_hint="sofa 옆",
                    location=MemoryLocation(name="living room"),
                ),
            ]
        )
        service = MemoryQueryService(repository)

        hits = service.search("user-1", "\ub0b4 \uc9c0\uac11 \uc5b4\ub514\uc5d0 \uc788\uc5c8\uc9c0?", 3)

        self.assertEqual(repository.last_user_id, "user-1")
        self.assertEqual(repository.last_limit, 40)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].memory.memory_id, "mem-wallet-01")
        self.assertIn("wallet", hits[0].matched_terms)

    def test_search_handles_korean_particle_query(self) -> None:
        repository = FakeMemoryRepository(
            [
                MemoryRecord(
                    memory_id="mem-umbrella-01",
                    user_id="user-2",
                    image_key="captures/umbrella-01.jpg",
                    image_url=None,
                    captured_at="2026-04-17T08:00:00Z",
                    caption="umbrella leaning against sofa",
                    scene_summary=None,
                    detected_objects=["umbrella", "sofa"],
                    tags=["living-room"],
                    ocr_text=None,
                    note=None,
                    position_hint="sofa 옆",
                    location=MemoryLocation(name="living room"),
                ),
            ]
        )
        service = MemoryQueryService(repository)

        hits = service.search("user-2", "\uc6b0\uc0b0\uc740 \uc5b4\ub514\uc5d0?", 3)

        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].memory.memory_id, "mem-umbrella-01")
        self.assertIn("umbrella", hits[0].matched_terms)

    def test_chat_falls_back_to_template_when_no_hits(self) -> None:
        service = MemoryQueryService(FakeMemoryRepository([]))

        answer, hits = service.chat("user-3", "\uc9c0\uac11 \uc5b4\ub514 \uc788\uc5c8\uc5b4?", 3)

        self.assertEqual(hits, [])
        self.assertEqual(answer.mode, "template")
        self.assertEqual(answer.cited_memory_ids, [])
        self.assertIn("\uba54\ubaa8\ub9ac", answer.text)

    def test_check_health_delegates_to_repository(self) -> None:
        repository = FakeMemoryRepository([])
        service = MemoryQueryService(repository)

        service.check_health()

        self.assertTrue(repository.health_checked)


if __name__ == "__main__":
    unittest.main()
