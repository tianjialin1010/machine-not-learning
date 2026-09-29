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


class ProviderError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


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


def _parse_jev_answer(identifier: str, answer: dict[str, Any]) -> Prediction:
    if not isinstance(answer, dict):
        raise ProviderError(f"invalid answer for {identifier}: expected object", retryable=False)
    answer_type = str(answer.get("type", ""))
    probabilities = {str(key): float(value) for key, value in dict(answer.get("probabilities", {})).items()}
    confidence = answer.get("confidence")
    if confidence is not None and (not math.isfinite(float(confidence)) or not 0 <= float(confidence) <= 1):
        raise ProviderError(f"invalid confidence for {identifier}", retryable=False)
    if answer_type == "choice" or "choice" in answer:
        if not probabilities or any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities.values()):
            raise ProviderError(f"invalid choice probabilities for {identifier}", retryable=False)
        if abs(sum(probabilities.values()) - 1) > 0.01:
            raise ProviderError(f"choice probabilities do not sum to one for {identifier}", retryable=False)
        choice = answer.get("choice")
        if choice is None or str(choice) not in probabilities:
            raise ProviderError(f"choice is missing from probabilities for {identifier}", retryable=False)
        return Prediction(identifier, "choice", str(choice), probabilities, confidence, "jev")
    if answer_type == "score" or "score" in answer:
        if not probabilities or any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities.values()):
            raise ProviderError(f"invalid score probabilities for {identifier}", retryable=False)
        if abs(sum(probabilities.values()) - 1) > 0.01:
            raise ProviderError(f"score probabilities do not sum to one for {identifier}", retryable=False)
        score = float(answer.get("score", 0.0))
        if not math.isfinite(score) or score < 0 or score > max(0, len(probabilities) - 1):
            raise ProviderError(f"score is out of range for {identifier}", retryable=False)
        return Prediction(identifier, "score", score, probabilities, confidence, "jev")
    if "noul" not in answer and "yes_probability" not in answer:
        raise ProviderError(f"missing noul value for {identifier}", retryable=False)
    noul = float(answer.get("noul", answer.get("yes_probability")))
    if not math.isfinite(noul) or not 0 <= noul <= 1:
        raise ProviderError(f"noul probability is out of range for {identifier}", retryable=False)
    return Prediction(identifier, "noul", noul >= 0.5, {"yes": noul, "no": 1 - noul}, None, "jev")


def _validate_answer_set(raw_answers: dict[str, Any], provider: str, profile: QuestionProfile | None = None) -> list[Prediction]:
    try:
        predictions = [_parse_jev_answer(identifier, answer) for identifier, answer in raw_answers.items()]
    except (TypeError, ValueError, KeyError) as exc:
        raise ProviderError(f"invalid {provider} answer shape: {exc}", retryable=False) from exc
    profile = profile or default_question_profile()
    schema = profile.schema()
    expected = set(schema)
    if set(raw_answers) != expected:
        raise ProviderError(f"{provider} response contained unexpected or missing decision ids", retryable=False)
    allowed = {
        identifier: set(spec.get("options", []))
        for identifier, spec in schema.items()
        if spec.get("type") == "choice"
    }
    for prediction in predictions:
        if prediction.id in allowed and str(prediction.value) not in allowed[prediction.id]:
            raise ProviderError(f"{provider} returned an unsupported choice for {prediction.id}", retryable=False)
        prediction.provider = provider
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


@dataclass
class OpenSourceProvider:
    """JEV-like adapter for vLLM, Ollama, or another OpenAI-compatible local server."""

    endpoint: str = "http://127.0.0.1:8000/v1/chat/completions"
    model: str = "local-model"
    api_key: str = "EMPTY"
    timeout: float = 30.0
    max_tokens: int = 1024          # ollama 默认仅 128，多问嵌套 JSON 会被截断
    name: str = "open-source"
    questions: QuestionProfile = field(default_factory=default_question_profile)
    profile: DecisionHeadProfile = DecisionHeadProfile(name="open-source-raw", version="0.1", label_provenance="model-self-report")
    last_report: CalibrationReport | None = None

    def evaluate(self, state: dict[str, Any]) -> list[Prediction]:
        schema = json.dumps(self.questions.schema(), ensure_ascii=False, indent=2)
        system = (
            "You are a typed decision classifier inside a software system. Do not write a reply.\n"
            "Return JSON only (no markdown fences) with a top-level 'answers' object, one entry per "
            "question id, using EXACTLY these shapes and key names:\n"
            '  - choice: {"type":"choice","choice":"<one of the declared options>",'
            '"probabilities":{"<option>":<p>,...}}\n'
            '  - score:  {"type":"score","score":<number>,"probabilities":'
            '{"<level>":<p>,...}}\n'
            '  - noul:   {"type":"noul","noul":<0.0-1.0>}\n'
            "Rules: never emit a \'value\' key; every probabilities object must sum to 1; "
            "the chosen option/score must be one of the declared ones; "
            "answer EVERY question id listed in the schema.\n"
            'Example output: {"answers":{"needs_evidence":{"type":"noul","noul":0.8}}}\n\n'
            f"Question schema:\n{schema}"
        )
        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(state, ensure_ascii=False)},
            ],
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"]
            if isinstance(content, list):
                content = "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
            parsed = _parse_json_object(str(content))
            raw_answers = parsed.get("answers", parsed)
            if not isinstance(raw_answers, dict):
                raise ProviderError("open-source response did not contain an answers object", retryable=True)
            raw_answers = _normalize_raw_answers(raw_answers)
            predictions = _validate_answer_set(raw_answers, self.name, self.questions)
            for prediction in predictions:
                prediction.calibration_status = "uncalibrated"
            self.last_report = apply_profile(predictions, self.profile, state)
            return predictions
        except ProviderError:
            raise
        except urllib.error.HTTPError as exc:
            raise ProviderError(f"open-source provider returned HTTP {exc.code}", exc.code, exc.code >= 500 or exc.code == 429) from exc
        except (urllib.error.URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError) as exc:
            raise ProviderError(f"open-source provider request failed: {exc}", retryable=True) from exc


@dataclass
class StructuredHeadProvider:
    """Adapter for a trained MLP/torch/sklearn decision head and a custom QuestionProfile."""

    evaluator: Any
    profile: DecisionHeadProfile
    questions: QuestionProfile = field(default_factory=default_question_profile)
    name: str = "structured-head"
    last_report: CalibrationReport | None = None

    def evaluate(self, state: dict[str, Any]) -> list[Prediction]:
        features = state.get("conversation_state", {}).get("structured_features")
        if not isinstance(features, dict) or not features:
            raise ProviderError("structured head requires conversation_state.structured_features", retryable=False)
        try:
            raw_answers = self.evaluator(features)
        except Exception as exc:
            raise ProviderError(f"structured head evaluation failed: {exc}", retryable=False) from exc
        if not isinstance(raw_answers, dict):
            raise ProviderError("structured head evaluator must return an answers object", retryable=False)
        predictions = _validate_answer_set(raw_answers, self.name, self.questions)
        self.last_report = apply_profile(predictions, self.profile, state)
        for prediction in predictions:
            prediction.provider = self.name
            if self.profile.calibrated:
                prediction.calibration_status = "temperature_scaled"
        return predictions


def _normalize_raw_answers(raw_answers: dict[str, Any]) -> dict[str, Any]:
    """把模型输出清洗成校验器能接受的形状（只做容错，不改语义）。

    真机实测（Qwen3-8B via ollama）暴露四类格式瑕疵，任何一类都会让整单请求
    以 retryable=False 硬失败：value 键名、概率和不为 1、score 越界、概率非数值。
    """
    for answer in raw_answers.values():
        if not isinstance(answer, dict):
            continue
        answer_type = str(answer.get("type", ""))
        probabilities = answer.get("probabilities")
        if not isinstance(probabilities, dict) or not probabilities:
            continue

        if "value" in answer:
            if answer_type == "choice" and "choice" not in answer:
                answer["choice"] = answer["value"]
            elif answer_type == "score" and "score" not in answer:
                answer["score"] = answer["value"]

        try:
            values = {str(key): float(value) for key, value in probabilities.items()}
        except (TypeError, ValueError):
            continue
        if any(not math.isfinite(value) or value < 0 for value in values.values()):
            continue

        total = sum(values.values())
        if total > 0 and abs(total - 1.0) > 1e-9:
            values = {key: value / total for key, value in values.items()}
        answer["probabilities"] = values

        if answer_type == "score" or "score" in answer:
            raw_score = answer.get("score")
            if isinstance(raw_score, str) and raw_score in values:
                raw_score = list(values).index(raw_score)
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(score):
                continue
            answer["score"] = min(max(score, 0.0), float(max(0, len(values) - 1)))

    return raw_answers


def _parse_json_object(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
        cleaned = cleaned.rsplit("```", 1)[0].strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, dict):
        raise json.JSONDecodeError("expected JSON object", cleaned, 0)
    return parsed


@dataclass
class TieredProvider:
    """Fast structured head first; escalate ambiguous or untrusted cases to an open model."""

    fast: DecisionProvider
    generative: DecisionProvider | None = None
    name: str = "tiered"
    fast_confidence_floor: float = 0.8
    last_route: str = "fast"
    last_reason: str = ""
    last_report: CalibrationReport | None = None
    degraded: bool = False

    def evaluate(self, state: dict[str, Any]) -> list[Prediction]:
        self.degraded = False
        fast_predictions = self.fast.evaluate(state)
        self.last_report = getattr(self.fast, "last_report", None)
        force_generative = bool(state.get("force_generative"))
        low_confidence = any(
            prediction.confidence is not None and prediction.confidence < self.fast_confidence_floor
            for prediction in fast_predictions
            if prediction.id in {"intent", "next_action", "risk_level"}
        )
        if self.generative and (force_generative or low_confidence):
            try:
                predictions = self.generative.evaluate(state)
                self.last_route = "generative"
                self.last_reason = "forced" if force_generative else "fast_head_low_confidence"
                self.last_report = getattr(self.generative, "last_report", None)
                return predictions
            except ProviderError:
                self.degraded = True
                self.last_route = "fast_degraded"
                self.last_reason = "generative_provider_failed"
        self.last_route = "fast"
        self.last_reason = "fast_head_accepted"
        return fast_predictions


@dataclass
class ResilientProvider:
    primary: DecisionProvider
    fallback: DecisionProvider
    name: str = "resilient"
    degraded: bool = False
    last_error: str | None = None

    def evaluate(self, state: dict[str, Any]) -> list[Prediction]:
        try:
            self.degraded = False
            self.last_error = None
            return self.primary.evaluate(state)
        except ProviderError as exc:
            self.degraded = True
            self.last_error = str(exc)
            if not exc.retryable:
                raise
            return self.fallback.evaluate(state)


def build_provider() -> DecisionProvider:
    mode = os.getenv("JEV_PROVIDER", "rules").lower()
    rules = RuleBasedProvider()
    if mode in {"opensource", "open-source", "local", "vllm", "ollama"}:
        endpoint = os.getenv("OPEN_SOURCE_ENDPOINT", "http://127.0.0.1:8000/v1/chat/completions")
        model = os.getenv("OPEN_SOURCE_MODEL", "local-model")
        api_key = os.getenv("OPEN_SOURCE_API_KEY", "EMPTY")
        return ResilientProvider(
            OpenSourceProvider(endpoint=endpoint, model=model, api_key=api_key),
            rules,
        )
    if mode == "tiered":
        endpoint = os.getenv("OPEN_SOURCE_ENDPOINT", "http://127.0.0.1:8000/v1/chat/completions")
        model = os.getenv("OPEN_SOURCE_MODEL", "local-model")
        generative = OpenSourceProvider(endpoint=endpoint, model=model, api_key=os.getenv("OPEN_SOURCE_API_KEY", "EMPTY"))
        return TieredProvider(rules, generative)
    if mode != "jev":
        return rules
    api_key = os.getenv("TYPESAFE_API_KEY", "").strip()
    if not api_key:
        raise ProviderError("JEV_PROVIDER=jev requires TYPESAFE_API_KEY")
    return ResilientProvider(JevProvider(api_key=api_key), rules)
