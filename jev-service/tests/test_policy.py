import unittest

from jev_service.models import Prediction
from jev_service.policy import PolicyConfig, choose_action


class PolicyTests(unittest.TestCase):
    def test_high_risk_is_blocked(self) -> None:
        result = choose_action([], "我想伤害自己", 0, "rules", PolicyConfig(high_risk_terms=("伤害自己",)))
        self.assertEqual(result.kind, "human_review")
        self.assertTrue(result.blocked)
        self.assertTrue(result.requires_approval)

    def test_history_is_retrieved_before_answer(self) -> None:
        predictions = [
            Prediction("needs_history", "noul", True, {"yes": 0.9, "no": 0.1}),
            Prediction("needs_clarification", "noul", False, {"yes": 0.1, "no": 0.9}),
            Prediction("next_action", "choice", "answer_from_context", {"answer_from_context": 0.9, "ask_clarification": 0.1}, 0.9),
        ]
        result = choose_action(predictions, "你还记得吗？", 0, "rules")
        self.assertEqual(result.kind, "retrieve_evidence")

    def test_ambiguous_choice_clarifies(self) -> None:
        predictions = [
            Prediction("needs_history", "noul", False, {"yes": 0.1, "no": 0.9}),
            Prediction("needs_clarification", "noul", False, {"yes": 0.1, "no": 0.9}),
            Prediction("next_action", "choice", "answer_from_context", {"answer_from_context": 0.45, "ask_clarification": 0.4, "draft_suggestion": 0.15}, 0.1),
        ]
        result = choose_action(predictions, "这件事", 1, "rules")
        self.assertEqual(result.kind, "ask_clarification")

    def test_low_confidence_choice_clarifies_even_with_a_clear_top_probability(self) -> None:
        predictions = [
            Prediction("next_action", "choice", "answer_from_context", {"answer_from_context": 0.9, "ask_clarification": 0.1}, 0.1),
        ]
        result = choose_action(predictions, "回答", 1, "rules")
        self.assertEqual(result.kind, "ask_clarification")

    def test_provider_review_is_never_downgraded_by_low_confidence(self) -> None:
        predictions = [
            Prediction("next_action", "choice", "human_review", {"human_review": 0.72, "ask_clarification": 0.28}, 0.1),
        ]
        result = choose_action(predictions, "请谨慎处理", 0, "rules")
        self.assertEqual(result.kind, "human_review")
        self.assertTrue(result.blocked)
        self.assertTrue(result.requires_approval)
