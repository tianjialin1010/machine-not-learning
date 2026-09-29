from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
import time
import urllib.error
import urllib.request
from typing import Any, Protocol

from .models import Prediction
from .questions import QuestionProfile, default_question_profile, question_schema
from .calibration import CalibrationReport, DecisionHeadProfile, apply_profile


from .errors import ProviderError
from .normalization import _format_error, _normalize_raw_answers, _parse_jev_answer, _validate_answer_set


class DecisionProvider(Protocol):
    name: str

    def evaluate(self, state: dict[str, Any]) -> list[Prediction]: ...


def _normalise_distribution(values: dict[str, float]) -> dict[str, float]:
    clipped = {key: max(0.0, float(value)) for key, value in values.items()}
    total = sum(clipped.values()) or 1.0
    return {key: round(value / total, 6) for key, value in clipped.items()}


def _confidence(probabilities: dict[str, float]) -> float:
    values = list(probabilities.values())
    if len(values) <= 1:
        return 1.0
    entropy = -sum(p * math.log(p, 2) for p in values if p > 0)
    maximum = math.log(len(values), 2)
    return round(max(0.0, min(1.0, 1 - entropy / maximum)), 4)


def _choice(identifier: str, probabilities: dict[str, float], provider: str) -> Prediction:
    probabilities = _normalise_distribution(probabilities)
    selected = max(probabilities, key=probabilities.get)
    return Prediction(
        id=identifier,
        type="choice",
        value=selected,
        probabilities=probabilities,
        confidence=_confidence(probabilities),
        provider=provider,
    )


def _noul(identifier: str, yes_probability: float, provider: str) -> Prediction:
    probability = round(max(0.0, min(1.0, yes_probability)), 6)
    return Prediction(
        id=identifier,
        type="noul",
        value=probability >= 0.5,
        probabilities={"yes": probability, "no": round(1 - probability, 6)},
        confidence=None,
        provider=provider,
    )


def _score(identifier: str, probabilities: dict[str, float], provider: str) -> Prediction:
    probabilities = _normalise_distribution(probabilities)
    if all(key.isdigit() for key in probabilities):
        ordered = sorted(probabilities.items(), key=lambda item: int(item[0]))
    else:
        # Named levels are defined in the question's insertion order.
        ordered = list(probabilities.items())
    score = sum(float(index) * probability for index, (_, probability) in enumerate(ordered))
    return Prediction(
        id=identifier,
        type="score",
        value=round(score, 4),
        probabilities=probabilities,
        confidence=_confidence(probabilities),
        provider=provider,
    )


@dataclass
class RuleBasedProvider:
    """Offline baseline only; replace term lists with labeled domain data before production."""

    name: str = "rules"
    evidence_terms: tuple[str, ...] = ("记得", "忘", "昨天", "之前", "上次", "说过", "previous", "before", "earlier", "history", "source", "evidence")
    planning_terms: tuple[str, ...] = ("安排", "计划", "时间", "怎么做", "帮我", "plan", "schedule", "how do", "help me")
    risk_terms: tuple[str, ...] = ("自杀", "伤害自己", "不想活", "暴力", "威胁", "紧急", "泄露", "越权", "删除全部", "支付", "credential", "breach", "urgent", "delete all")
    data_change_terms: tuple[str, ...] = ("删除", "修改", "更新", "写入", "发送", "delete", "change", "update", "write", "send")
    failure_terms: tuple[str, ...] = ("失败", "错误", "无法", "broken", "failed", "error", "not working")
    profile: DecisionHeadProfile = DecisionHeadProfile(name="rules-baseline", version="0.1", label_provenance="heuristic")
    last_report: CalibrationReport | None = None

    def evaluate(self, state: dict[str, Any]) -> list[Prediction]:
        message = str(state.get("current_message", ""))
        lowered = message.lower()
        has_question = "?" in message or "吗" in message or "是不是" in message
        history_terms = self.evidence_terms
        planning_terms = self.planning_terms
        urgent_terms = self.risk_terms
        failure_terms = self.failure_terms
        action_terms = self.data_change_terms + ("请你", "帮我")

        if any(term in message for term in urgent_terms):
            intent = _choice("intent", {"safety_or_compliance": 0.55, "data_change": 0.1, "other": 0.35}, self.name)
            risk = _score("risk_level", {"low": 0.01, "moderate": 0.04, "high": 0.2, "critical": 0.75}, self.name)
        elif any(term in message.lower() for term in failure_terms):
            intent = _choice("intent", {"safety_or_compliance": 0.2, "information_request": 0.32, "other": 0.38, "task_execution": 0.1}, self.name)
            risk = _score("risk_level", {"low": 0.05, "moderate": 0.25, "high": 0.6, "critical": 0.1}, self.name)
        elif any(term in message for term in planning_terms) or any(term in message for term in action_terms):
            intent = _choice("intent", {"task_execution": 0.52, "planning": 0.3, "information_request": 0.1, "other": 0.08}, self.name)
            risk = _score("risk_level", {"low": 0.62, "moderate": 0.3, "high": 0.07, "critical": 0.01}, self.name)
        elif has_question:
            intent = _choice("intent", {"information_request": 0.64, "clarification": 0.2, "planning": 0.08, "other": 0.08}, self.name)
            risk = _score("risk_level", {"low": 0.72, "moderate": 0.22, "high": 0.05, "critical": 0.01}, self.name)
        else:
            intent = _choice("intent", {"other": 0.48, "planning": 0.2, "information_request": 0.18, "clarification": 0.14}, self.name)
            risk = _score("risk_level", {"low": 0.5, "moderate": 0.35, "high": 0.13, "critical": 0.02}, self.name)

        history_probability = 0.86 if any(term in message for term in history_terms) else 0.18
        clarification_probability = 0.76 if len(message.strip()) < 8 or message.strip() in {"所以呢", "那你说", "然后呢"} else 0.2
        side_effect_probability = 0.72 if "我授权" in message or "直接发送" in message else 0.08
        if any(term in message for term in urgent_terms):
            next_action = _choice("next_action", {"human_review": 0.72, "ask_clarification": 0.12, "stop": 0.1, "draft_suggestion": 0.06}, self.name)
        elif clarification_probability >= 0.7:
            next_action = _choice("next_action", {"ask_clarification": 0.68, "retrieve_evidence": 0.18, "draft_suggestion": 0.08, "answer_from_context": 0.06}, self.name)
        elif history_probability >= 0.7:
            next_action = _choice("next_action", {"retrieve_evidence": 0.72, "ask_clarification": 0.12, "draft_suggestion": 0.1, "answer_from_context": 0.06}, self.name)
        elif any(term in message for term in action_terms):
            next_action = _choice("next_action", {"draft_suggestion": 0.56, "ask_clarification": 0.18, "answer_from_context": 0.16, "human_review": 0.1}, self.name)
        else:
            next_action = _choice("next_action", {"answer_from_context": 0.58, "ask_clarification": 0.2, "draft_suggestion": 0.14, "retrieve_evidence": 0.08}, self.name)

        predictions = [
            intent,
            risk,
            _noul("needs_evidence", history_probability, self.name),
            _noul("needs_clarification", clarification_probability, self.name),
            _noul("allows_side_effect", side_effect_probability, self.name),
            next_action,
        ]
        for prediction in predictions:
            prediction.calibration_status = "heuristic"
        self.last_report = apply_profile(predictions, self.profile, state)
        return predictions


@dataclass
class JevProvider:
    api_key: str
    model: str = "jev-latest"
    endpoint: str = "https://api.typesafe.ai/v1/systemone"
    timeout: float = 8.0
    name: str = "jev"
    questions: QuestionProfile = field(default_factory=default_question_profile)

    def evaluate(self, state: dict[str, Any]) -> list[Prediction]:
        payload = {"model": self.model, "state": state, "questions": self.questions.schema()}
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        last_error: ProviderError | None = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                retryable = exc.code in {429, 529} or exc.code >= 500
                last_error = ProviderError(f"jev provider returned HTTP {exc.code}", exc.code, retryable)
                if not retryable or attempt == 2:
                    raise last_error from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = ProviderError(f"jev provider request failed: {exc}", retryable=True)
                if attempt == 2:
                    raise last_error from exc
            except json.JSONDecodeError as exc:
                raise ProviderError(f"invalid JSON from jev provider: {exc}", retryable=False) from exc
            time.sleep(0.2 * (2**attempt))
        else:
            raise last_error or ProviderError("jev provider request failed", retryable=True)

        raw_answers = body.get("answers", body.get("responses", {}))
        if isinstance(raw_answers, list):
            raw_answers = {str(item.get("id")): item for item in raw_answers}
        if not isinstance(raw_answers, dict):
            raise ProviderError("jev response did not contain answers", retryable=False)
        predictions = _validate_answer_set(raw_answers, "jev", self.questions)
        for prediction in predictions:
            prediction.calibration_status = "provider_calibrated"
        return predictions


from .http_provider import OpenSourceProvider, StepFunProvider, _parse_json_object


@dataclass
class StructuredHeadProvider:
    """Adapter for a trained MLP/torch/sklearn decision head and a custom QuestionProfile."""

    evaluator: Any
    profile: DecisionHeadProfile
    questions: QuestionProfile = field(default_factory=default_question_profile)
    name: str = "structured-head"
    last_report: CalibrationReport | None = None

    def __post_init__(self) -> None:
        schema = self.questions.schema()
        if set(self.profile.temperatures) - set(schema):
            raise ValueError("temperature profile references unknown questions")
        if self.profile.prior_correction_head and (self.profile.prior_correction_head not in schema or schema[self.profile.prior_correction_head]["type"] != "noul"):
            raise ValueError("prior correction must bind to a declared Noul question")
        if self.profile.class_support_head and self.profile.class_support_head not in schema:
            raise ValueError("class support references an unknown question")

    def accepts(self, state: dict[str, Any]) -> bool:
        try:
            self._features(state)
        except ProviderError:
            return False
        return True

    def _features(self, state: dict[str, Any]) -> dict[str, float]:
        context = state.get("conversation_state", {})
        features = context.get("structured_features")
        if not isinstance(features, dict) or not features:
            raise ProviderError("structured head requires nonempty features", code="invalid_features")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in features.values()):
            raise ProviderError("structured features must be finite numbers", code="invalid_features")
        if self.profile.feature_count is not None and len(features) != self.profile.feature_count:
            raise ProviderError("feature dimension does not match profile", code="invalid_features")
        if self.profile.feature_names and set(features) != set(self.profile.feature_names):
            raise ProviderError("feature names do not match profile", code="invalid_features")
        for key, (low, high) in self.profile.feature_bounds.items():
            if key not in features or not low <= features[key] <= high:
                raise ProviderError("feature outside declared bounds", code="invalid_features")
        for key, unit in self.profile.feature_units.items():
            units = context.get("feature_units", {})
            if not isinstance(units, dict) or units.get(key) != unit:
                raise ProviderError("feature unit does not match profile", code="invalid_features")
        return {key: features[key] for key in (self.profile.feature_names or features)}

    def evaluate(self, state: dict[str, Any]) -> list[Prediction]:
        self.last_report = None
        features = self._features(state)
        try:
            raw_answers = self.evaluator(features)
        except Exception as exc:
            raise ProviderError(f"structured head evaluation failed: {exc}", retryable=False) from exc
        if not isinstance(raw_answers, dict):
            raise ProviderError("structured head evaluator must return an answers object", retryable=False)
        predictions = _validate_answer_set(raw_answers, self.name, self.questions)
        for prediction in predictions:
            prediction.calibration_status = "uncalibrated"
        self.last_report = apply_profile(predictions, self.profile, state)
        for prediction in predictions:
            prediction.provider = self.name
        return predictions


from .routing import AbstainProvider, ResilientProvider, TieredProvider


def build_provider() -> DecisionProvider:
    mode = os.getenv("JEV_PROVIDER", "rules").lower()
    rules = RuleBasedProvider()
    profile_path = os.getenv("JEV_QUESTION_PROFILE")
    questions = QuestionProfile.from_file(profile_path) if profile_path else default_question_profile()
    if mode == "stepfun":
        base = os.getenv("STEPFUN_BASE_URL", "https://api.stepfun.com/v1").rstrip("/")
        primary = StepFunProvider(
            endpoint=base + "/chat/completions",
            model=os.getenv("STEPFUN_MODEL", "step-3.5-flash"),
            api_key=os.getenv("STEPFUN_API_KEY", ""),
            timeout=float(os.getenv("STEPFUN_TIMEOUT", "30")),
            max_tokens=int(os.getenv("STEPFUN_MAX_TOKENS", "4096")),
            max_retries=int(os.getenv("STEPFUN_MAX_RETRIES", "2")),
            response_format=os.getenv("STEPFUN_RESPONSE_FORMAT", "json_schema"),
            questions=questions,
        )
        return ResilientProvider(primary, AbstainProvider(questions))
    if mode in {"opensource", "open-source", "local", "vllm", "ollama"}:
        endpoint = os.getenv("OPEN_SOURCE_ENDPOINT", "http://127.0.0.1:8000/v1/chat/completions")
        model = os.getenv("OPEN_SOURCE_MODEL", "local-model")
        api_key = os.getenv("OPEN_SOURCE_API_KEY", "EMPTY")
        return ResilientProvider(
            OpenSourceProvider(endpoint=endpoint, model=model, api_key=api_key, questions=questions),
            AbstainProvider(questions) if profile_path else rules,
        )
    if mode == "tiered":
        if profile_path:
            raise ProviderError("custom tiered profiles require a programmatically configured structured head", code="configuration_error")
        endpoint = os.getenv("OPEN_SOURCE_ENDPOINT", "http://127.0.0.1:8000/v1/chat/completions")
        model = os.getenv("OPEN_SOURCE_MODEL", "local-model")
        generative = OpenSourceProvider(endpoint=endpoint, model=model, api_key=os.getenv("OPEN_SOURCE_API_KEY", "EMPTY"))
        return TieredProvider(rules, generative)
    if mode == "rules":
        if profile_path:
            raise ProviderError("rules baseline does not support custom question profiles", code="configuration_error")
        return rules
    if mode != "jev":
        raise ProviderError("unknown JEV_PROVIDER", code="configuration_error")
    api_key = os.getenv("TYPESAFE_API_KEY", "").strip()
    if not api_key:
        raise ProviderError("JEV_PROVIDER=jev requires TYPESAFE_API_KEY")
    return ResilientProvider(JevProvider(api_key=api_key, questions=questions), AbstainProvider(questions) if profile_path else rules)
