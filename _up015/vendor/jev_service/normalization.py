"""Strict typed decisions with a small, auditable compatibility layer."""
from __future__ import annotations

import math
from typing import Any
from .errors import ProviderError
from .models import Prediction
from .questions import QuestionProfile, default_question_profile


def _format_error(message: str) -> ProviderError:
    return ProviderError(message, retryable=True, code="invalid_output")


def number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _format_error(f"{label} must be numeric")
    try:
        value = float(value)
    except (OverflowError, ValueError) as exc:
        raise _format_error(f"{label} must be finite") from exc
    if not math.isfinite(value):
        raise _format_error(f"{label} must be finite")
    return value


def _parse_jev_answer(identifier: str, answer: dict[str, Any]) -> Prediction:
    if not isinstance(answer, dict):
        raise _format_error(f"invalid answer: {identifier}")
    kind = answer.get("type")
    if kind not in {"choice", "score", "noul"}:
        raise _format_error(f"invalid type: {identifier}")
    confidence = answer.get("confidence")
    if confidence is not None:
        confidence = number(confidence, "confidence")
        if not 0 <= confidence <= 1:
            raise _format_error("confidence outside 0..1")
    if kind == "noul":
        yes = number(answer.get("noul", answer.get("yes_probability")), "noul")
        if not 0 <= yes <= 1:
            raise _format_error("noul outside 0..1")
        return Prediction(identifier, kind, yes >= 0.5, {"yes": yes, "no": 1 - yes}, None, "jev")
    raw = answer.get("probabilities")
    if not isinstance(raw, dict) or not raw:
        raise _format_error("missing probabilities")
    probabilities = {key: number(value, "probability") for key, value in raw.items()}
    if any(not isinstance(k, str) for k in probabilities) or any(not 0 <= p <= 1 for p in probabilities.values()):
        raise _format_error("invalid probability distribution")
    if not math.isclose(math.fsum(probabilities.values()), 1.0, abs_tol=1e-5):
        raise _format_error("probabilities do not sum to one")
    if kind == "choice":
        value = answer.get("choice")
        if not isinstance(value, str) or value not in probabilities:
            raise _format_error("choice missing from distribution")
    else:
        value = number(answer.get("score"), "score")
        if not 0 <= value <= len(probabilities) - 1:
            raise _format_error("score out of range")
    return Prediction(identifier, kind, value, probabilities, confidence, "jev")


def _normalize_raw_answers(raw_answers: dict[str, Any], profile: QuestionProfile) -> tuple[dict[str, Any], list[str]]:
    schema = profile.schema()
    if not isinstance(raw_answers, dict) or set(raw_answers) != set(schema):
        raise _format_error("answers contained missing or unexpected question ids")
    result, warnings = {}, []
    for identifier, spec in schema.items():
        if not isinstance(raw_answers[identifier], dict):
            raise _format_error(f"invalid answer: {identifier}")
        item = dict(raw_answers[identifier])
        kind = spec["type"]
        if item.get("type", kind) != kind:
            raise _format_error(f"question type mismatch: {identifier}")
        item["type"] = kind
        native = {"choice": "choice", "score": "score", "noul": "noul"}[kind]
        if kind == "noul" and "yes_probability" in item:
            if "noul" in item and item["noul"] != item["yes_probability"]:
                raise _format_error("conflicting noul aliases")
            item["noul"] = item.pop("yes_probability")
        if "value" in item:
            alias = item.pop("value")
            if native in item and item[native] != alias:
                raise _format_error("conflicting value alias")
            if native not in item:
                item[native] = alias
                warnings.append(f"{identifier}:value_to_{native}")
        allowed = {"type", native, "confidence"} | ({"probabilities"} if kind != "noul" else set())
        if set(item) - allowed:
            raise _format_error(f"unexpected answer fields: {identifier}")
        if kind == "noul":
            if isinstance(item.get(native), bool):
                item[native] = float(item[native])
                warnings.append(f"{identifier}:boolean_to_probability")
            item[native] = number(item.get(native), native)
        else:
            raw_probs = item.get("probabilities")
            if not isinstance(raw_probs, dict) or not raw_probs or any(not isinstance(k, str) for k in raw_probs):
                raise _format_error("probabilities must be an object with string labels")
            probs = {k: number(v, "probability") for k, v in raw_probs.items()}
            if any(v < 0 for v in probs.values()) or max(probs.values()) <= 0:
                raise _format_error("negative or zero-sum probabilities")
            if kind == "score":
                levels = spec["levels"]
                zero = [str(i) for i in range(len(levels))]
                one = [str(i + 1) for i in range(len(levels))]
                if set(probs) == set(levels):
                    probs = {str(i): probs[label] for i, label in enumerate(levels)}
                    warnings.append(f"{identifier}:score_labels_to_indices")
                elif profile.one_based_score and set(probs) == set(one):
                    probs = {str(i): probs[str(i + 1)] for i in range(len(levels))}
                    warnings.append(f"{identifier}:one_based_probabilities_to_zero_based")
                elif set(probs) != set(zero):
                    raise _format_error("Score distribution must contain every declared level")
                probs = {key: probs[key] for key in zero}
                value = item.get("score")
                if isinstance(value, str) and value in levels:
                    value = float(levels.index(value))
                    warnings.append(f"{identifier}:score_label_to_index")
                else:
                    value = number(value, "score")
                    if profile.one_based_score:
                        if not 1 <= value <= len(levels):
                            raise _format_error("one-based Score out of range")
                        value -= 1
                        warnings.append(f"{identifier}:one_based_score_to_zero_based")
                item["score"] = value
            else:
                labels = spec["options"]
                if set(probs) - set(labels):
                    raise _format_error("unknown Choice distribution label")
                if set(probs) != set(labels):
                    warnings.append(f"{identifier}:missing_choice_probabilities_filled_zero")
                probs = {label: probs.get(label, 0.0) for label in labels}
            # Divide by max before summation to avoid finite-input overflow.
            maximum = max(probs.values())
            scaled = {k: v / maximum for k, v in probs.items()}
            total = math.fsum(scaled.values())
            if not math.isclose(maximum * total, 1.0, abs_tol=1e-6):
                warnings.append(f"{identifier}:probabilities_normalized")
            item["probabilities"] = {k: v / total for k, v in scaled.items()}
        _parse_jev_answer(identifier, item)
        result[identifier] = item
    return result, warnings


def _validate_answer_set(raw_answers: dict[str, Any], provider: str, profile: QuestionProfile | None = None) -> list[Prediction]:
    profile = profile or default_question_profile()
    if not isinstance(raw_answers, dict) or set(raw_answers) != set(profile.schema()):
        raise _format_error("missing or unexpected decision ids")
    predictions = []
    for identifier, spec in profile.schema().items():
        answer = raw_answers[identifier]
        if not isinstance(answer, dict):
            raise _format_error("answer must be an object")
        answer = dict(answer)
        answer.setdefault("type", spec["type"])
        prediction = _parse_jev_answer(identifier, answer)
        if prediction.type != spec["type"]:
            raise _format_error("question type mismatch")
        if prediction.type == "choice" and (prediction.value not in spec["options"] or set(prediction.probabilities) - set(spec["options"])):
            raise _format_error("unsupported Choice label")
        if prediction.type == "score":
            labels = [str(i) for i in range(len(spec["levels"]))]
            probs = prediction.probabilities
            if set(probs) == set(spec["levels"]):
                probs = {str(i): probs[label] for i, label in enumerate(spec["levels"])}
            if set(probs) != set(labels) or prediction.value > len(labels) - 1:
                raise _format_error("invalid Score labels or range")
            prediction.probabilities = {label: probs[label] for label in labels}
        prediction.provider = provider
        prediction.raw_value = prediction.value
        prediction.raw_probabilities = dict(prediction.probabilities)
        predictions.append(prediction)
    return predictions
