from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .memory import Evidence
from .models import DecisionRequest


@dataclass(frozen=True)
class StateBuildReport:
    input_modalities: tuple[str, ...]
    feature_status: str
    warnings: list[str]
    routing_hint: str


class StateBuilder:
    """Build a provider-neutral state and describe what kind of input it contains."""

    def build(self, request: DecisionRequest, evidence: list[Evidence]) -> tuple[dict[str, Any], StateBuildReport]:
        structured_features = request.conversation_state.get("structured_features")
        warnings: list[str] = []
        modalities = ["text"] if request.current_message else []
        feature_status = "absent"
        routing_hint = "generative"
        if structured_features is not None:
            modalities.append("structured_features")
            if isinstance(structured_features, dict) and structured_features:
                numeric = all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in structured_features.values())
                if numeric:
                    feature_status = "valid"
                    routing_hint = "fast"
                else:
                    feature_status = "invalid"
                    warnings.append("structured_features_non_numeric")
            else:
                feature_status = "invalid"
                warnings.append("structured_features_must_be_non_empty_object")
        else:
            warnings.append("no_structured_features")

        state = {
            "conversation_id": request.conversation_id,
            "current_message": request.current_message,
            "recent_turns": [turn.__dict__ for turn in request.recent_turns[-12:]],
            "retrieved_evidence": [item.to_dict() for item in evidence],
            "conversation_state": request.conversation_state,
            "user_policy": request.user_policy,
            "channel_metadata": request.channel_metadata,
            "input_profile": {
                "modalities": modalities,
                "feature_status": feature_status,
                "routing_hint": routing_hint,
            },
            "force_generative": routing_hint == "generative" and bool(request.user_policy.get("prefer_generative")),
            "distribution_id": request.channel_metadata.get("distribution_id"),
            "deployment_positive_rate": request.channel_metadata.get("deployment_positive_rate"),
        }
        return state, StateBuildReport(tuple(modalities), feature_status, warnings, routing_hint)
