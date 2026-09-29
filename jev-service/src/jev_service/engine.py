from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any

from .composer import TemplateComposer
from .memory import MemoryService
from .models import DecisionRequest, DecisionResponse, MemoryWriteProposal, new_id
from .policy import PolicyConfig, choose_action
from .provider import DecisionProvider, ResilientProvider, RuleBasedProvider, build_provider


@dataclass
class TraceRecord:
    trace_id: str
    conversation_id: str
    provider: str
    degraded: bool
    event: dict[str, Any]


class DecisionEngine:
    def __init__(
        self,
        provider: DecisionProvider | None = None,
        memory: MemoryService | None = None,
        policy: PolicyConfig | None = None,
        composer: TemplateComposer | None = None,
    ) -> None:
        self.provider = provider or build_provider()
        self.memory = memory or MemoryService()
        self.policy = policy or PolicyConfig()
        self.composer = composer or TemplateComposer()
        self.traces: list[TraceRecord] = []
        self._processed_events: dict[str, tuple[str, DecisionResponse]] = {}
        self._event_lock = threading.RLock()

    @property
    def degraded(self) -> bool:
        return bool(getattr(self.provider, "degraded", False))

    @property
    def provider_name(self) -> str:
        primary = getattr(self.provider, "primary", None)
        return getattr(primary, "name", getattr(self.provider, "name", "unknown"))

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        with self._event_lock:
            return self._decide(request)

    def _decide(self, request: DecisionRequest) -> DecisionResponse:
        trace_id = new_id("trace")
        if request.event_id and request.event_id in self._processed_events:
            fingerprint, original = self._processed_events[request.event_id]
            current_fingerprint = f"{request.conversation_id}\x00{request.current_message}"
            if fingerprint != current_fingerprint:
                raise ValueError("event_id was already used for a different conversation or message")
            replayed = DecisionResponse(
                trace_id=original.trace_id,
                conversation_id=original.conversation_id,
                decisions=original.decisions,
                recommended_action=original.recommended_action,
                evidence_refs=original.evidence_refs,
                memory_write_proposal=original.memory_write_proposal,
                draft_reply=original.draft_reply,
                degraded=original.degraded,
                deduplicated=True,
                created_at=original.created_at,
            )
            return replayed
        evidence = request.retrieved_evidence or self.memory.retrieve(request.conversation_id, request.current_message)
        state = {
            "conversation_id": request.conversation_id,
            "current_message": request.current_message,
            "recent_turns": [turn.__dict__ for turn in request.recent_turns[-12:]],
            "retrieved_evidence": [item.to_dict() for item in evidence],
            "conversation_state": request.conversation_state,
            "user_policy": request.user_policy,
            "channel_metadata": request.channel_metadata,
        }
        decisions = self.provider.evaluate(state)
        provider_status = "degraded" if self.degraded else self.provider_name
        action = choose_action(decisions, request.current_message, len(evidence), provider_status, self.policy)
        proposal: MemoryWriteProposal | None = None
        if request.user_policy.get("memory_write_content"):
            proposal = self.memory.propose_write(
                request.conversation_id,
                str(request.user_policy["memory_write_content"]),
                record_type=str(request.user_policy.get("memory_type", "semantic")),
                source=str(request.user_policy.get("memory_source", "inference")),
                confidence=float(request.user_policy.get("memory_confidence", 0.0)),
            )
        response = DecisionResponse(
            trace_id=trace_id,
            conversation_id=request.conversation_id,
            decisions=decisions,
            recommended_action=action,
            evidence_refs=[item.id for item in evidence],
            memory_write_proposal=proposal,
            draft_reply=self.composer.compose(request, action, evidence),
            degraded=self.degraded,
        )
        # Append after retrieval so the current event cannot masquerade as historical evidence.
        self.memory.append_event(request.conversation_id, request.current_message)
        if request.event_id:
            self._processed_events[request.event_id] = (
                f"{request.conversation_id}\x00{request.current_message}",
                response,
            )
        self.traces.append(
            TraceRecord(
                trace_id=trace_id,
                conversation_id=request.conversation_id,
                provider=provider_status,
                degraded=self.degraded,
                event={
                    "decisions": response.to_dict()["decisions"],
                    "action": action.to_dict(),
                    "evidence_refs": response.evidence_refs,
                },
            )
        )
        return response

    def replay(self, requests: list[DecisionRequest]) -> dict[str, Any]:
        responses = [self.decide(request) for request in requests]
        actions: dict[str, int] = {}
        for response in responses:
            kind = response.recommended_action.kind
            actions[kind] = actions.get(kind, 0) + 1
        return {
            "count": len(responses),
            "actions": actions,
            "degraded": any(response.degraded for response in responses),
            "results": [response.to_dict() for response in responses],
        }
