from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

QUESTIONS_VERSION = "dialogue-router.v1"


@dataclass(frozen=True)
class QuestionProfile:
    name: str
    version: str
    questions: dict[str, dict[str, Any]] = field(default_factory=dict)
    one_based_score: bool = False
    roles: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.one_based_score, bool) or not isinstance(self.roles, dict):
            raise ValueError("one_based_score must be boolean and roles must be an object")
        if not self.name or not self.version or not isinstance(self.questions, dict) or not self.questions:
            raise ValueError("QuestionProfile requires a name, version and nonempty questions")
        for identifier, spec in self.questions.items():
            if not isinstance(identifier, str) or not identifier or not isinstance(spec, dict):
                raise ValueError("invalid question definition")
            kind = spec.get("type")
            if kind not in {"choice", "score", "noul"}:
                raise ValueError(f"unsupported question type: {identifier}")
            if kind in {"choice", "score"}:
                values = spec.get("options" if kind == "choice" else "levels")
                if not isinstance(values, list) or len(values) < 2 or any(not isinstance(x, str) or not x for x in values) or len(set(values)) != len(values):
                    raise ValueError(f"invalid labels: {identifier}")
        expected = {"risk": "score", "action": "choice", "evidence": "noul", "clarification": "noul", "side_effect": "noul"}
        for role, identifier in self.roles.items():
            if role not in expected or identifier not in self.questions or self.questions[identifier]["type"] != expected[role]:
                raise ValueError(f"invalid question role: {role}")

    def schema(self) -> dict[str, dict[str, Any]]:
        return self.questions

    def answer_schema(self) -> dict[str, Any]:
        """JSON Schema sent to compatible providers; semantic checks still run locally."""
        def obj(properties: dict[str, Any]) -> dict[str, Any]:
            return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}
        probability = {"type": "number", "minimum": 0, "maximum": 1}
        answers = {}
        for identifier, spec in self.questions.items():
            kind = spec["type"]
            fields: dict[str, Any] = {"type": {"type": "string", "enum": [kind]}}
            if kind == "noul":
                fields["noul"] = probability
            else:
                if kind == "choice":
                    labels = spec["options"]
                    fields["choice"] = {"type": "string", "enum": labels}
                else:
                    start = 1 if self.one_based_score else 0
                    labels = [str(i + start) for i in range(len(spec["levels"]))]
                    fields["score"] = {"type": "number", "minimum": start, "maximum": start + len(labels) - 1}
                fields["probabilities"] = obj({label: probability for label in labels})
                fields["confidence"] = {"type": ["number", "null"], "minimum": 0, "maximum": 1}
            answers[identifier] = obj(fields)
        return obj({"answers": obj(answers)})

    @classmethod
    def from_file(cls, path: str) -> "QuestionProfile":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        allowed = {"name", "version", "questions", "one_based_score", "roles"}
        if not isinstance(value, dict) or set(value) - allowed:
            raise ValueError("invalid QuestionProfile file")
        return cls(**value)


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
