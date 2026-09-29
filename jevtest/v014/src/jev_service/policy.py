from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .models import ActionRecommendation, Prediction


@dataclass(frozen=True)
class PolicyConfig:
    auto_confidence: float = 0.80
    auto_probability: float = 0.70
    ambiguity_margin: float = 0.20
    urgent_score: float = 2.0
    # Empty by default: a deployed domain supplies its own lexical risk policy.
    high_risk_terms: tuple[str, ...] = ()
    require_calibration_for_auto: bool = True
    task_probability_thresholds: dict[str, float] = field(default_factory=dict)
    ood_review_threshold: float = 0.70
    unknown_values: tuple[str, ...] = ("unknown", "abstain")


def _find(predictions: Iterable[Prediction], identifier: str) -> Prediction | None:
    return next((prediction for prediction in predictions if prediction.id == identifier), None)


def _top_probability(prediction: Prediction | None) -> float:
    return max(prediction.probabilities.values()) if prediction and prediction.probabilities else 0.0


def _margin(prediction: Prediction | None) -> float:
    if not prediction or not prediction.probabilities:
        return 0.0
    values = sorted(prediction.probabilities.values(), reverse=True)
    return values[0] - (values[1] if len(values) > 1 else 0.0)


def choose_action(
    predictions: list[Prediction],
    current_message: str,
    evidence_count: int,
    provider_status: str,
    config: PolicyConfig | None = None,
    warnings: list[str] | None = None,
) -> ActionRecommendation:
    config = config or PolicyConfig()
    reasons: list[str] = []
    warnings = warnings or []
    tension = _find(predictions, "risk_level") or _find(predictions, "tension")
    next_action = _find(predictions, "next_action")
    needs_history = _find(predictions, "needs_evidence") or _find(predictions, "needs_history")
    needs_clarification = _find(predictions, "needs_clarification")
    allows_side_effect = _find(predictions, "allows_side_effect")

    if "out_of_distribution" in warnings:
        reasons.append("out_of_distribution_requires_review")
        return ActionRecommendation("human_review", True, reasons, None, provider_status, True)
    for prediction in predictions:
        rare_warnings = [warning for warning in prediction.warnings if warning.startswith("rare_classes:")]
        if not rare_warnings:
            continue
        rare_labels = {
            label
            for warning in rare_warnings
            for label in warning.split(":", 1)[1].split(",")
            if label
        }
        predicted_label = prediction.value if prediction.type == "choice" else None
        if prediction.type == "score" and prediction.probabilities:
            predicted_label = max(prediction.probabilities, key=prediction.probabilities.get)
        if predicted_label in rare_labels:
            reasons.append("rare_class_support_insufficient")
            return ActionRecommendation("human_review", True, reasons, None, provider_status, True)
    if any(isinstance(prediction.value, str) and prediction.value in config.unknown_values for prediction in predictions):
        reasons.append("prediction_abstained_or_unknown")
        return ActionRecommendation("abstain", False, reasons, None, provider_status, True)

    high_risk = any(term.lower() in current_message.lower() for term in config.high_risk_terms)
    if high_risk or (tension and isinstance(tension.value, (float, int)) and float(tension.value) >= config.urgent_score):
        reasons.append("high_risk_or_urgent")
        return ActionRecommendation("human_review", True, reasons, tension.confidence if tension else None, provider_status, True)

    if "distribution_shift" in warnings:
        reasons.append("distribution_shift_requires_recalibration")
        return ActionRecommendation("human_review", True, reasons, None, provider_status, True)

    if allows_side_effect and allows_side_effect.probabilities.get("yes", 0.0) < config.auto_probability:
        reasons.append("side_effect_not_authorized")

    clarification_threshold = config.task_probability_thresholds.get("needs_clarification", config.auto_probability)
    if needs_clarification and needs_clarification.probabilities.get("yes", 0.0) >= clarification_threshold:
        reasons.append("clarification_needed")
        return ActionRecommendation("ask_clarification", False, reasons, needs_clarification.probabilities.get("yes"), provider_status)

    evidence_threshold = config.task_probability_thresholds.get("needs_evidence", config.auto_probability)
    if needs_history and needs_history.probabilities.get("yes", 0.0) >= evidence_threshold and evidence_count == 0:
        reasons.append("history_evidence_missing")
        return ActionRecommendation("retrieve_evidence", False, reasons, needs_history.probabilities.get("yes"), provider_status)

    candidate = next_action.value if next_action and isinstance(next_action.value, str) else "ask_clarification"
    candidate_confidence = next_action.confidence if next_action else None
    if candidate == "human_review":
        reasons.append("provider_requested_review")
        return ActionRecommendation("human_review", True, reasons, candidate_confidence, provider_status, True)
    if config.require_calibration_for_auto and any(
        prediction.calibration_status == "uncalibrated" for prediction in predictions
    ):
        reasons.append("uncalibrated_probability")
        return ActionRecommendation("ask_clarification", False, reasons, candidate_confidence, provider_status)
    next_action_threshold = config.task_probability_thresholds.get("next_action", config.auto_probability)
    if next_action and (
        _top_probability(next_action) < next_action_threshold
        or _margin(next_action) < config.ambiguity_margin
        or (next_action.confidence is not None and next_action.confidence < config.auto_confidence)
    ):
        reasons.append("low_decision_confidence")
        return ActionRecommendation("ask_clarification", False, reasons, candidate_confidence, provider_status)

    if candidate in {"draft_suggestion", "answer_from_context"} and evidence_count == 0 and needs_history and needs_history.value:
        reasons.append("candidate_needs_evidence")
        return ActionRecommendation("retrieve_evidence", False, reasons, candidate_confidence, provider_status)

    reasons.append("policy_route")
    return ActionRecommendation(candidate, False, reasons, candidate_confidence, provider_status)
