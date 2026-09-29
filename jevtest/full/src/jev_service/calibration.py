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


def _calibrate_probability(probability: float, profile: DecisionHeadProfile, deploy_rate: float | None) -> float:
    probability = min(1 - 1e-9, max(1e-9, probability))
    logit = math.log(probability / (1 - probability))
    if profile.train_positive_rate and deploy_rate and 0 < deploy_rate < 1:
        train = min(1 - 1e-9, max(1e-9, profile.train_positive_rate))
        logit += math.log((deploy_rate / (1 - deploy_rate)) / (train / (1 - train)))
    logit /= max(profile.temperature, 1e-6)
    return 1 / (1 + math.exp(-logit))


def apply_profile(
    predictions: list[Prediction],
    profile: DecisionHeadProfile,
    state: dict[str, Any] | None = None,
) -> CalibrationReport:
    """Apply optional calibration metadata and annotate distribution/data limitations."""
    state = state or {}
    warnings: list[str] = []
    deploy_rate = state.get("deployment_positive_rate")
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
        if prediction.type == "noul":
            yes = float(prediction.probabilities.get("yes", 0.5))
            if profile.calibrated:
                yes = _calibrate_probability(yes, profile, float(deploy_rate) if deploy_rate is not None else None)
                prediction.probabilities = {"yes": round(yes, 6), "no": round(1 - yes, 6)}
                prediction.value = yes >= 0.5
                prediction.calibration_status = "temperature_scaled"
        elif profile.calibrated and prediction.probabilities:
            keys = list(prediction.probabilities)
            logits = [math.log(max(prediction.probabilities[key], 1e-9)) for key in keys]
            scaled = _softmax([value / max(profile.temperature, 1e-6) for value in logits])
            prediction.probabilities = {key: round(value, 6) for key, value in zip(keys, scaled)}
            # 注意：只有 choice 类型才把 value 设为概率最高的**标签名**。
            # score 类型的 value 必须保持数值——policy 用
            # isinstance(tension.value, (float, int)) 判断紧急度，
            # 一旦被改写成字符串，紧急度保护会静默失效。
            if prediction.type == "choice":
                prediction.value = max(prediction.probabilities,
                                       key=prediction.probabilities.get)
            elif prediction.type == "score":
                best = max(range(len(keys)), key=lambda index: scaled[index])
                prediction.value = float(best)
            prediction.calibration_status = "temperature_scaled"
        if "rare_classes" in " ".join(warnings) and prediction.id == "intent":
            prediction.calibration_status = "limited_support"

    status = "calibrated" if profile.calibrated and not shifted else "shifted" if shifted else "uncalibrated"
    return CalibrationReport(profile=f"{profile.name}@{profile.version}", status=status, warnings=warnings, distribution_shifted=shifted)
