from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
import uuid


DecisionType = Literal["choice", "score", "noul"]
ActionKind = Literal[
    "answer_from_context",
    "retrieve_evidence",
    "retrieve_history",
    "ask_clarification",
    "draft_suggestion",
    "human_review",
    "stop",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


@dataclass(frozen=True)
class Turn:
    role: Literal["user", "assistant", "system", "tool"]
    content: str
    timestamp: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Turn":
        role = str(value.get("role", "user"))
        if role not in {"user", "assistant", "system", "tool"}:
            raise ValueError(f"unsupported turn role: {role}")
        content = str(value.get("content", ""))
        if not content:
            raise ValueError("turn content cannot be empty")
        return cls(role=role, content=content, timestamp=value.get("timestamp"))


@dataclass(frozen=True)
class Evidence:
    id: str
    content: str
    source: str = "memory"
    score: float = 0.0
    occurred_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DecisionRequest:
    conversation_id: str
    current_message: str
    event_id: str | None = None
    recent_turns: list[Turn] = field(default_factory=list)
    retrieved_evidence: list[Evidence] = field(default_factory=list)
    conversation_state: dict[str, Any] = field(default_factory=dict)
    user_policy: dict[str, Any] = field(default_factory=dict)
    channel_metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DecisionRequest":
        if not isinstance(value, dict):
            raise ValueError("request body must be an object")
        conversation_id = str(value.get("conversation_id", "")).strip()
        current_message = str(value.get("current_message", "")).strip()
        if not conversation_id:
            raise ValueError("conversation_id is required")
        if not current_message:
            raise ValueError("current_message is required")
        raw_turns = value.get("recent_turns", [])
        if raw_turns is None or not isinstance(raw_turns, list):
            raise ValueError("recent_turns must be an array")
        if any(not isinstance(item, dict) for item in raw_turns):
            raise ValueError("each recent_turns item must be an object")
        turns = [Turn.from_dict(item) for item in raw_turns]
        raw_evidence = value.get("retrieved_evidence", [])
        if raw_evidence is None or not isinstance(raw_evidence, list):
            raise ValueError("retrieved_evidence must be an array")
        if any(not isinstance(item, dict) for item in raw_evidence):
            raise ValueError("each retrieved_evidence item must be an object")
        evidence = [
            Evidence(
                id=str(item.get("id", new_id("evidence"))),
                content=str(item.get("content", "")),
                source=str(item.get("source", "memory")),
                score=float(item.get("score", 0.0)),
                occurred_at=item.get("occurred_at"),
            )
            for item in raw_evidence
            if str(item.get("content", "")).strip()
        ]
        return cls(
            conversation_id=conversation_id,
            current_message=current_message,
            event_id=str(value["event_id"]) if value.get("event_id") else None,
            recent_turns=turns,
            retrieved_evidence=evidence,
            conversation_state=_mapping(value.get("conversation_state", {}), "conversation_state"),
            user_policy=_mapping(value.get("user_policy", {}), "user_policy"),
            channel_metadata=_mapping(value.get("channel_metadata", {}), "channel_metadata"),
        )


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None or not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return dict(value)


@dataclass
class Prediction:
    id: str
    type: DecisionType
    value: str | float | bool
    probabilities: dict[str, float]
    confidence: float | None = None
    provider: str = "rules"
    calibration_status: str = "unknown"
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ActionRecommendation:
    kind: ActionKind
    requires_approval: bool
    reason_codes: list[str]
    confidence: float | None
    provider_status: str
    blocked: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryWriteProposal:
    content: str
    record_type: Literal["semantic", "procedural"]
    source: str
    eligible: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DecisionResponse:
    trace_id: str
    conversation_id: str
    decisions: list[Prediction]
    recommended_action: ActionRecommendation
    evidence_refs: list[str]
    memory_write_proposal: MemoryWriteProposal | None
    draft_reply: str
    degraded: bool
    deduplicated: bool = False
    route: str = "fast"
    calibration_profile: str = "unconfigured@0"
    warnings: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "conversation_id": self.conversation_id,
            "decisions": [item.to_dict() for item in self.decisions],
            "recommended_action": self.recommended_action.to_dict(),
            "evidence_refs": self.evidence_refs,
            "memory_write_proposal": self.memory_write_proposal.to_dict()
            if self.memory_write_proposal
            else None,
            "draft_reply": self.draft_reply,
            "degraded": self.degraded,
            "deduplicated": self.deduplicated,
            "route": self.route,
            "calibration_profile": self.calibration_profile,
            "warnings": self.warnings,
            "created_at": self.created_at,
        }
