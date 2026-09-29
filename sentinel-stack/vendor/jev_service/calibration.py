from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

from .models import Prediction


@dataclass(frozen=True)
class DecisionHeadProfile:
    """Metadata and calibration policy for a structured decision head."""

    name: str = "unconfigured"
    version: str = "0"
    label_provenance: str = "unknown"
    calibrated: bool = False
    calibration_distribution: str = "unknown"
    temperature: float = 1.0
    train_positive_rate: float | None = None
    min_class_support: int = 0
    class_support: dict[str, int] = field(default_factory=dict)
    rare_class_threshold: int = 10
    class_support_head: str | None = None
    temperatures: dict[str, float] = field(default_factory=dict)
    prior_correction_head: str | None = None
    calibration_positive_rate: float | None = None
    prior_shift_assumption: str | None = None
    feature_count: int | None = None
    feature_names: tuple[str, ...] = ()
    feature_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    feature_units: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for temperature in [self.temperature, *self.temperatures.values()]:
            if isinstance(temperature, bool) or not math.isfinite(temperature) or temperature < 1e-6:
                raise ValueError("temperature must be finite and >= 1e-6")
        for rate in [self.train_positive_rate, self.calibration_positive_rate]:
            if rate is not None and (isinstance(rate, bool) or not math.isfinite(rate) or not 0 < rate < 1):
                raise ValueError("source positive rate must be strictly between zero and one")
        if self.feature_count is not None and (isinstance(self.feature_count, bool) or not isinstance(self.feature_count, int) or self.feature_count < 1):
            raise ValueError("feature_count must be a positive integer")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must be unique")
        if self.feature_count and self.feature_names and len(self.feature_names) != self.feature_count:
            raise ValueError("feature_count and feature_names disagree")
        for bounds in self.feature_bounds.values():
            if len(bounds) != 2 or not all(math.isfinite(x) for x in bounds) or bounds[0] > bounds[1]:
                raise ValueError("invalid feature bounds")


@dataclass
class CalibrationReport:
    profile: str
    status: str
    warnings: list[str]
    distribution_shifted: bool = False


def _softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    exps = [math.exp(value - maximum) for value in values]
    total = sum(exps) or 1.0
    return [value / total for value in exps]


def _calibrate_probability(probability: float, profile: DecisionHeadProfile, deploy_rate: float | None, temperature: float | None = None) -> float:
    probability = min(1 - 1e-9, max(1e-9, probability))
    logit = math.log(probability / (1 - probability)) / (temperature or profile.temperature)
    source = profile.calibration_positive_rate if profile.calibration_positive_rate is not None else profile.train_positive_rate
    if source is not None and deploy_rate is not None:
        # Temperature calibration first, then prior-odds correction under label shift.
        logit += math.log(deploy_rate / (1 - deploy_rate)) - math.log(source / (1 - source))
    if logit >= 0:
        return 1 / (1 + math.exp(-logit))
    exp_value = math.exp(logit)
    return exp_value / (1 + exp_value)


def apply_profile(
    predictions: list[Prediction],
    profile: DecisionHeadProfile,
    state: dict[str, Any] | None = None,
) -> CalibrationReport:
    """Apply optional calibration metadata and annotate distribution/data limitations."""
    state = state or {}
    warnings: list[str] = []
    deploy_rate = state.get("deployment_positive_rate")
    if deploy_rate is not None and (isinstance(deploy_rate, bool) or not isinstance(deploy_rate, (int, float)) or not math.isfinite(deploy_rate) or not 0 < deploy_rate < 1):
        warnings.append("invalid_deployment_prior")
        deploy_rate = None
    correction_enabled = bool(profile.prior_correction_head and profile.prior_shift_assumption == "label_shift" and (profile.calibration_positive_rate is not None or profile.train_positive_rate is not None))
    if deploy_rate is not None and not correction_enabled:
        warnings.append("prior_correction_not_configured")
    shifted = bool(state.get("distribution_id") and state.get("distribution_id") != profile.calibration_distribution)
    if shifted:
        warnings.append("distribution_shift")
    if profile.label_provenance in {"derived_rule", "synthetic_only", "unknown"}:
        warnings.append(f"label_provenance:{profile.label_provenance}")
    if profile.min_class_support and profile.class_support:
        rare = [label for label, support in profile.class_support.items() if support < profile.rare_class_threshold]
        if rare:
            warnings.append("rare_classes:" + ",".join(sorted(rare)))

    for prediction in predictions:
        if prediction.raw_value is None:
            prediction.raw_value = prediction.value
            prediction.raw_probabilities = dict(prediction.probabilities)
        temperature = profile.temperatures.get(prediction.id, profile.temperature)
        if prediction.type == "noul":
            yes = float(prediction.probabilities.get("yes", 0.5))
            if profile.calibrated:
                target_rate = deploy_rate if correction_enabled and prediction.id == profile.prior_correction_head else None
                yes = _calibrate_probability(yes, profile, target_rate, temperature)
                if target_rate is not None:
                    prediction.warnings.append("label_shift_prior_corrected")
                prediction.probabilities = {"yes": round(yes, 6), "no": round(1 - yes, 6)}
                prediction.value = yes >= 0.5
                prediction.calibration_status = "temperature_scaled"
        elif profile.calibrated and prediction.probabilities:
            keys = list(prediction.probabilities)
            if prediction.type == "score":
                if set(keys) != {str(i) for i in range(len(keys))}:
                    raise ValueError("calibrated Score requires canonical numeric level keys")
                keys = sorted(keys, key=int)
            logits = [math.log(max(prediction.probabilities[key], 1e-9)) for key in keys]
            scaled = _softmax([value / temperature for value in logits])
            prediction.probabilities = {key: round(value, 6) for key, value in zip(keys, scaled)}
            if prediction.type == "choice":
                prediction.value = max(prediction.probabilities, key=prediction.probabilities.get)
            elif prediction.type == "score":
                prediction.value = round(sum(index * probability for index, probability in enumerate(scaled)), 6)
            prediction.calibration_status = "temperature_scaled"
        if profile.class_support_head and prediction.id == profile.class_support_head:
            rare = [label for label, support in profile.class_support.items() if support < profile.rare_class_threshold]
            if rare:
                warning = "rare_classes:" + ",".join(sorted(rare))
                prediction.warnings.append(warning)
                # Limited class support is independent of whether probabilities are calibrated.
        if shifted:
            prediction.calibration_status = "distribution_shifted"

    status = "calibrated" if profile.calibrated and not shifted else "shifted" if shifted else "uncalibrated"
    return CalibrationReport(profile=f"{profile.name}@{profile.version}", status=status, warnings=warnings, distribution_shifted=shifted)
