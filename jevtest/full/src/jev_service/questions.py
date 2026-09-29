from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

QUESTIONS_VERSION = "dialogue-router.v1"


@dataclass(frozen=True)
class QuestionProfile:
    name: str
    version: str
    questions: dict[str, dict[str, Any]] = field(default_factory=dict)

    def schema(self) -> dict[str, dict[str, Any]]:
        return self.questions


def question_schema() -> dict[str, dict[str, object]]:
    """Return domain-neutral atomic questions for the default router profile."""
    return default_question_profile().schema()


def default_question_profile() -> QuestionProfile:
    return QuestionProfile(
        name="dialogue-router",
        version=QUESTIONS_VERSION,
        questions={
        "intent": {
            "type": "choice",
            "question": "What is the primary intent of the current message?",
            "options": [
                "information_request",
                "clarification",
                "planning",
                "task_execution",
                "data_change",
                "safety_or_compliance",
                "other",
            ],
        },
        "risk_level": {
            "type": "score",
            "question": "What is the operational risk level of taking the next action?",
            "levels": ["low", "moderate", "high", "critical"],
        },
        "needs_evidence": {
            "type": "noul",
            "question": "Does answering safely require retrieving additional evidence?",
        },
        "needs_clarification": {
            "type": "noul",
            "question": "Is the message ambiguous enough that clarification should come before an answer?",
        },
        "allows_side_effect": {
            "type": "noul",
            "question": "Has the user clearly authorized an external side-effect action?",
        },
        "next_action": {
            "type": "choice",
            "question": "What should the application do next?",
            "options": [
                "answer_from_context",
                "retrieve_evidence",
                "ask_clarification",
                "draft_suggestion",
                "human_review",
                "stop",
            ],
        },
        },
    )
