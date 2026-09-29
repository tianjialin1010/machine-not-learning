import json
import unittest
from unittest.mock import patch

from jev_service.provider import OpenSourceProvider, ProviderError, ResilientProvider, RuleBasedProvider, _parse_jev_answer

from jev_service.provider import RuleBasedProvider


class RuleProviderTests(unittest.TestCase):
    def test_history_message_returns_atomic_decisions(self) -> None:
        predictions = RuleBasedProvider().evaluate({"current_message": "你还记得我昨天说过的事吗？"})
        self.assertEqual({item.id for item in predictions}, {
            "intent", "risk_level", "needs_evidence", "needs_clarification", "allows_side_effect", "next_action"
        })
        history = next(item for item in predictions if item.id == "needs_evidence")
        self.assertTrue(history.value)
        self.assertAlmostEqual(sum(history.probabilities.values()), 1.0, places=5)

    def test_urgent_message_is_review_candidate(self) -> None:
        predictions = RuleBasedProvider().evaluate({"current_message": "我不想活了，感觉很紧急"})
        next_action = next(item for item in predictions if item.id == "next_action")
        self.assertEqual(next_action.value, "human_review")

    def test_remote_answer_validation_rejects_invalid_probability(self) -> None:
        with self.assertRaises(ProviderError):
            _parse_jev_answer("needs_history", {"type": "noul", "noul": 2})

    def test_non_retryable_provider_error_is_not_silently_downgraded(self) -> None:
        class Broken:
            name = "broken"
            def evaluate(self, state):
                raise ProviderError("unauthorized", 401, retryable=False)

        with self.assertRaises(ProviderError):
            ResilientProvider(Broken(), RuleBasedProvider()).evaluate({"current_message": "hi"})

    def test_open_source_provider_uses_typed_json_contract(self) -> None:
        answers = {
            "intent": {"type": "choice", "choice": "information_request", "probabilities": {"information_request": 1}},
            "risk_level": {"type": "score", "score": 0, "probabilities": {"0": 1, "1": 0, "2": 0, "3": 0}},
            "needs_evidence": {"type": "noul", "noul": 0.1},
            "needs_clarification": {"type": "noul", "noul": 0.1},
            "allows_side_effect": {"type": "noul", "noul": 0.1},
            "next_action": {"type": "choice", "choice": "answer_from_context", "probabilities": {"answer_from_context": 1}},
        }

        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): return None
            def read(self):
                return json.dumps({"choices": [{"message": {"content": json.dumps({"answers": answers})}}]}).encode()

        with patch("jev_service.provider.urllib.request.urlopen", return_value=Response()):
            predictions = OpenSourceProvider().evaluate({"current_message": "hello"})
        self.assertEqual(len(predictions), 6)
        self.assertTrue(all(item.provider == "open-source" for item in predictions))
        self.assertTrue(all(item.calibration_status == "uncalibrated" for item in predictions))
