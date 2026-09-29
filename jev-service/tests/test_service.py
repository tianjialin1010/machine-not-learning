import unittest

from jev_service.engine import DecisionEngine
from jev_service.service import Service


class ServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = Service(DecisionEngine())

    def test_decide_returns_trace_and_draft(self) -> None:
        status, payload = self.service.handle(
            "/v1/decide",
            {"conversation_id": "c1", "current_message": "你还记得昨天说过的事吗？"},
        )
        self.assertEqual(status, 200)
        self.assertTrue(payload["trace_id"].startswith("trace_"))
        self.assertTrue(payload["draft_reply"])
        self.assertIn("recommended_action", payload)

    def test_replay_and_health(self) -> None:
        status, payload = self.service.handle(
            "/v1/replay",
            {"requests": [
                {"conversation_id": "c1", "current_message": "帮我安排时间"},
                {"conversation_id": "c1", "current_message": "我不想活了"},
            ]},
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["count"], 2)
        self.assertIn("human_review", payload["actions"])
        status, health = self.service.handle("/healthz", {})
        self.assertEqual(status, 200)
        self.assertTrue(health["ok"])

    def test_invalid_request_is_rejected(self) -> None:
        status, payload = self.service.handle("/v1/decide", {"current_message": "缺会话 id"})
        self.assertEqual(status, 422)
        self.assertIn("conversation_id", payload["error"])
        status, payload = self.service.handle(
            "/v1/decide", {"conversation_id": "c1", "current_message": "x", "recent_turns": None}
        )
        self.assertEqual(status, 422)
        self.assertIn("recent_turns", payload["error"])

    def test_duplicate_event_is_deduplicated(self) -> None:
        request = {"conversation_id": "c1", "event_id": "msg-1", "current_message": "你好"}
        first_status, first = self.service.handle("/v1/decide", request)
        second_status, second = self.service.handle("/v1/decide", request)
        self.assertEqual((first_status, second_status), (200, 200))
        self.assertFalse(first["deduplicated"])
        self.assertTrue(second["deduplicated"])
        self.assertEqual(first["trace_id"], second["trace_id"])
        status, payload = self.service.handle(
            "/v1/decide", {"conversation_id": "other", "event_id": "msg-1", "current_message": "different"}
        )
        self.assertEqual(status, 422)
        self.assertIn("different conversation", payload["error"])
