from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from .calibration import CalibrationReport
from .errors import ProviderError
from .models import Prediction
from .questions import QuestionProfile, default_question_profile


@dataclass
class AbstainProvider:
    questions: QuestionProfile = field(default_factory=default_question_profile)
    name: str = "abstain"
    last_route: str = "abstain"
    last_reason: str = "provider_unavailable"
    last_warnings: list[str] = field(default_factory=lambda: ["provider_unavailable"])
    last_report: CalibrationReport | None = None

    def evaluate(self, state) -> list[Prediction]:
        self.last_report = CalibrationReport("none", "unavailable", ["provider_unavailable"], False)
        return []


@dataclass
class ResilientProvider:
    primary: Any
    fallback: Any
    degraded: bool = False
    last_error: str | None = None
    last_report: CalibrationReport | None = None
    last_normalization_warnings: list[str] = field(default_factory=list)
    last_warnings: list[str] = field(default_factory=list)
    last_audit: list[dict[str, Any]] = field(default_factory=list)
    last_safety_predictions: list[Prediction] = field(default_factory=list)
    last_route: str = "fast"
    last_reason: str = ""

    @property
    def questions(self):
        return getattr(self.primary, "questions", default_question_profile())

    def _collect(self, provider):
        self.last_report = getattr(provider, "last_report", None)
        self.last_normalization_warnings.extend(getattr(provider, "last_normalization_warnings", []))
        self.last_warnings.extend(getattr(provider, "last_warnings", []))
        self.last_audit.extend(getattr(provider, "last_audit", []))
        self.last_safety_predictions.extend(getattr(provider, "last_safety_predictions", []))
        self.last_route = getattr(provider, "last_route", "fast")
        self.last_reason = getattr(provider, "last_reason", "provider_succeeded")

    def evaluate(self, state):
        self.degraded = False
        self.last_error = None
        self.last_report = None
        self.last_normalization_warnings, self.last_warnings, self.last_audit, self.last_safety_predictions = [], [], [], []
        try:
            result = self.primary.evaluate(state)
            self._collect(self.primary)
            return result
        except ProviderError as exc:
            self._collect(self.primary)
            self.last_error = exc.code
            if not exc.retryable:
                raise
            self.degraded = True
            self.last_warnings.append("provider_fallback:" + exc.code)
            result = self.fallback.evaluate(state)
            self._collect(self.fallback)
            self.last_reason = "fallback:" + exc.code
            return result


@dataclass
class TieredProvider:
    fast: Any
    generative: Any | None = None
    name: str = "tiered"
    fast_confidence_floor: float = 0.8
    last_route: str = "fast"
    last_reason: str = ""
    last_report: CalibrationReport | None = None
    degraded: bool = False
    last_normalization_warnings: list[str] = field(default_factory=list)
    last_warnings: list[str] = field(default_factory=list)
    last_audit: list[dict[str, Any]] = field(default_factory=list)
    last_safety_predictions: list[Prediction] = field(default_factory=list)

    def __post_init__(self):
        left = getattr(self.fast, "questions", None)
        right = getattr(self.generative, "questions", None)
        if left is not None and right is not None and (left.name, left.version, left.schema(), left.roles) != (right.name, right.version, right.schema(), right.roles):
            raise ValueError("tiered providers must share the same question profile")

    @property
    def questions(self):
        return getattr(self.fast, "questions", getattr(self.generative, "questions", default_question_profile()))

    def evaluate(self, state):
        self.degraded = False
        self.last_report = None
        self.last_normalization_warnings, self.last_warnings, self.last_audit, self.last_safety_predictions = [], [], [], []
        eligible = not hasattr(self.fast, "accepts") or self.fast.accepts(state)
        try:
            fast_predictions = self.fast.evaluate(state) if eligible else []
        except ProviderError as exc:
            if not exc.retryable:
                raise
            eligible = False
            fast_predictions = []
            self.degraded = True
            self.last_warnings.append("fast_provider_failed:" + exc.code)
        if eligible:
            self.last_safety_predictions = list(fast_predictions)
            self.last_report = getattr(self.fast, "last_report", None)
            self.last_warnings.extend(getattr(self.last_report, "warnings", []))
            self.last_normalization_warnings.extend(getattr(self.fast, "last_normalization_warnings", []))
            self.last_audit.extend(getattr(self.fast, "last_audit", []))
        low_confidence = any((p.confidence is not None and p.confidence < self.fast_confidence_floor) or p.calibration_status in {"uncalibrated", "unknown", "distribution_shifted"} for p in fast_predictions)
        escalate = not eligible or bool(state.get("force_generative")) or low_confidence
        if escalate and self.generative:
            try:
                predictions = self.generative.evaluate(state)
                self.last_audit.extend(getattr(self.generative, "last_audit", []))
                self.last_report = getattr(self.generative, "last_report", None)
                self.last_normalization_warnings.extend(getattr(self.generative, "last_normalization_warnings", []))
                self.last_warnings.extend(getattr(self.generative, "last_warnings", []))
                self.last_route = getattr(self.generative, "last_route", "generative")
                self.last_reason = "features_not_supported" if not eligible else "forced" if state.get("force_generative") else "fast_head_uncertain"
                before = {p.id:p for p in fast_predictions}
                for current in predictions:
                    old = before.get(current.id)
                    if old and (old.type != current.type or (current.type != "score" and old.value != current.value) or (current.type == "score" and abs(float(old.value) - float(current.value)) >= 1)):
                        self.last_warnings.append("provider_disagreement")
                return predictions
            except ProviderError as exc:
                self.last_audit.extend(getattr(self.generative, "last_audit", []))
                if not exc.retryable:
                    raise
                self.degraded = True
                self.last_warnings.append("provider_fallback:" + exc.code)
                self.last_route = "fast" if eligible else "abstain"
                self.last_reason = "generative_failed:" + exc.code
                if not eligible:
                    self.last_warnings.append("provider_unavailable")
                return fast_predictions
        if not eligible:
            self.last_route, self.last_reason = "abstain", "no_eligible_provider"
            self.last_warnings.append("provider_unavailable")
            return []
        self.last_route, self.last_reason = "fast", "fast_head_accepted"
        return fast_predictions
