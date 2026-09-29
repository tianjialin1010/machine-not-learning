"""Held-out dataset replay using real models, alongside the existing CNC service."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import threading
import time
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4


SCENARIOS = {
    "ai4i": ("数控机床", "fault_and_type"),
    "skab": ("水泵机组", "binary_anomaly"),
    "steel": ("钢板检测", "defect_classification"),
    "secom": ("晶圆工艺", "binary_quality"),
    "tep": ("化工过程", "process_classification"),
}
CNC_CLASSES = ["No Failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
CNC_FEATURES = [
    {"key": "air_temp_c", "label": "环境温度", "unit": "°C", "kind": "physical"},
    {"key": "process_temp_c", "label": "工艺温度", "unit": "°C", "kind": "physical"},
    {"key": "rpm", "label": "转速", "unit": "rpm", "kind": "physical"},
    {"key": "torque_nm", "label": "扭矩", "unit": "N·m", "kind": "physical"},
    {"key": "tool_wear_min", "label": "刀具使用时长", "unit": "min", "kind": "physical"},
]
CNC_UNITS = {"air_temp_c": "C", "process_temp_c": "C", "rpm": "rpm", "torque_nm": "Nm", "tool_wear_min": "min"}
MAX_BODY_BYTES = 64 * 1024


class IndustrialError(Exception):
    def __init__(self, status, code, message):
        super().__init__(message)
        self.status, self.code = status, code


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def display_values(descriptors, features):
    return [{**descriptor, "value": features[descriptor["key"]]} for descriptor in descriptors]


def validate_metadata(meta):
    if not isinstance(meta, dict):
        raise ValueError("metadata must be an object")
    names = meta.get("raw_feature_names")
    classes = meta.get("classes")
    if not isinstance(names, list) or not names or not all(isinstance(n, str) and n for n in names) or len(set(names)) != len(names):
        raise ValueError("raw_feature_names must be unique nonempty strings")
    if not isinstance(classes, list) or len(classes) < 2 or not all(isinstance(c, str) and c for c in classes) or len(set(classes)) != len(classes):
        raise ValueError("classes must contain at least two unique labels")
    if meta.get("normal_class") is not None and meta["normal_class"] not in classes:
        raise ValueError("normal_class must be an exported class or null")
    if not isinstance(meta.get("model_version"), str) or not meta["model_version"]:
        raise ValueError("model_version is required")
    if not isinstance(meta.get("model_sha256"), str) or len(meta["model_sha256"]) != 64:
        raise ValueError("model_sha256 is required")
    if not finite_number(meta.get("temperature")) or meta["temperature"] <= 0:
        raise ValueError("temperature must be finite and positive")
    if not isinstance(meta.get("calibrated"), bool):
        raise ValueError("calibrated must explicitly describe the exported model")
    indices = meta.get("selected_feature_indices")
    if not isinstance(indices, list) or not indices or any(type(i) is not int or not 0 <= i < len(names) for i in indices) or len(set(indices)) != len(indices):
        raise ValueError("selected_feature_indices are invalid")
    for key, length in (("imputer_medians", len(names)), ("scaler_mean", len(indices)), ("scaler_scale", len(indices))):
        values = meta.get(key)
        if not isinstance(values, list) or len(values) != length or not all(finite_number(v) for v in values):
            raise ValueError(key + " does not match the feature contract")
    if any(value <= 0 for value in meta["scaler_scale"]):
        raise ValueError("scaler_scale must be positive")
    descriptors = meta.get("display_features")
    if not isinstance(descriptors, list) or not descriptors:
        raise ValueError("display_features are required")
    for descriptor in descriptors:
        if not isinstance(descriptor, dict) or descriptor.get("key") not in names or not isinstance(descriptor.get("label"), str):
            raise ValueError("display_features must refer to original input fields")
        if descriptor.get("kind") not in ("physical", "model_feature") or not (descriptor.get("unit") is None or isinstance(descriptor["unit"], str)):
            raise ValueError("display_features units/kind are invalid")
    if not isinstance(meta.get("dataset"), dict) or not meta["dataset"].get("name") or not meta["dataset"].get("source"):
        raise ValueError("dataset provenance is required")


def validate_samples(samples, names, classes):
    if not isinstance(samples, list) or not samples:
        raise ValueError("held-out samples are missing")
    seen = set()
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("sample must be an object")
        identifier = sample.get("id")
        if not isinstance(identifier, str) or not identifier or len(identifier) > 200 or identifier in seen:
            raise ValueError("sample ids must be unique nonempty strings")
        seen.add(identifier)
        features = sample.get("features")
        if sample.get("split") != "test" or sample.get("expected_class") not in classes:
            raise ValueError("only labeled held-out test samples may be replayed")
        if not isinstance(features, dict) or set(features) != set(names):
            raise ValueError("sample fields do not match the feature contract")
        if any(value is not None and not finite_number(value) for value in features.values()):
            raise ValueError("sample values must be finite numbers or missing")


def normalized_features(meta, features):
    raw = [features[name] if features[name] is not None else meta["imputer_medians"][i]
           for i, name in enumerate(meta["raw_feature_names"])]
    result = [(raw[index] - mean) / scale for index, mean, scale in zip(
        meta["selected_feature_indices"], meta["scaler_mean"], meta["scaler_scale"])]
    if not all(finite_number(value) for value in result):
        raise ValueError("non-finite preprocessed input")
    return result


class IndustrialRuntime:
    def __init__(self, root, device):
        self.root = Path(root)
        self.artifacts = (self.root / "industrial-models-current").resolve()
        self.device = device
        self.entries = {}
        self.lock = threading.RLock()

    def _entry(self, scenario):
        with self.lock:
            if scenario in self.entries:
                return self.entries[scenario]
            directory = self.artifacts / scenario
            try:
                meta = json.loads((directory / "metadata.json").read_text())
                validate_metadata(meta)
                if meta.get("scenario") != scenario:
                    raise ValueError("metadata scenario does not match its directory")
                samples = json.loads((directory / "heldout_samples.json").read_text())
                validate_samples(samples, meta["raw_feature_names"], meta["classes"])
                entry = {"meta": meta, "samples": samples, "by_id": {s["id"]: s for s in samples}}
                self.entries[scenario] = entry
                return entry
            except (OSError, ValueError, TypeError, KeyError) as error:
                raise IndustrialError(503, "artifacts_unavailable", f"{scenario}: {error}") from error

    def _model(self, scenario):
        entry = self._entry(scenario)
        with self.lock:
            if "network" in entry:
                return entry
            try:
                import torch
                meta = entry["meta"]
                weights = self.artifacts / scenario / "model.pt"
                digest = file_sha256(weights)
                if digest != meta["model_sha256"]:
                    raise ValueError("model weight hash does not match metadata")
                network = torch.nn.Sequential(
                    torch.nn.Linear(len(meta["selected_feature_indices"]), 64), torch.nn.ReLU(),
                    torch.nn.Linear(64, 32), torch.nn.ReLU(),
                    torch.nn.Linear(32, len(meta["classes"])),
                )
                network.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
                network.to(self.device).eval()
                entry.update(network=network, weight_hash=digest, torch=torch,
                             actual_device=str(next(network.parameters()).device))
                return entry
            except Exception as error:
                raise IndustrialError(503, "model_unavailable", f"{scenario}: {error}") from error

    def _cnc_entry(self, handler):
        with self.lock:
            if "ai4i" in self.entries:
                return self.entries["ai4i"]
            try:
                import sentinel_heads as heads
                rows = heads.holdout_rows(handler.meta["data_path"])
                samples = [{"id": f"ai4i-test-{i}", "label": f"测试样本 {i + 1} · {row['type']}",
                            "features": {item["key"]: row[item["key"]] for item in CNC_FEATURES},
                            "split": "test", "expected_class": row["type"]}
                           for i, row in enumerate(rows)]
                validate_samples(samples, [item["key"] for item in CNC_FEATURES], CNC_CLASSES)
                entry = {"samples": samples, "by_id": {s["id"]: s for s in samples}, "meta": {
                    "raw_feature_names": [item["key"] for item in CNC_FEATURES],
                    "model_input_dim": len(heads.FEATURE_NAMES),
                    "display_features": CNC_FEATURES, "classes": CNC_CLASSES, "normal_class": "No Failure",
                    "dataset": {"name": "AI4I 2020", "source": "https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset", "split": "test", "split_method": "existing sentinel_heads.holdout_rows"},
                }}
                self.entries["ai4i"] = entry
                return entry
            except Exception as error:
                raise IndustrialError(503, "samples_unavailable", f"ai4i: {error}") from error

    def _cnc_model_info(self, handler):
        if handler.engine is None or handler.service is None:
            raise IndustrialError(503, "model_unavailable", "Existing CNC engine is not ready")
        try:
            digest = file_sha256(self.root / "heads.pt")
        except OSError as error:
            raise IndustrialError(503, "model_unavailable", "Existing CNC weights cannot be verified") from error
        return {"name": handler.meta["decision_head"], "version": handler.meta["framework"],
                "device": handler.meta["device"], "sha256": digest}

    def catalog(self, handler):
        scenarios = []
        for identifier, (label, task) in SCENARIOS.items():
            item = {"id": identifier, "label": label, "task": task, "available": False,
                    "status": "unavailable", "classes": [], "normal_class": None,
                    "feature_count": 0, "model_input_dim": None, "metrics": None,
                    "display_features": [], "model": None, "dataset": None,
                    "limitations": ["数据集留出样本回放，不是实时产线遥测。", "分类结果不代表三维模型中的部件定位。"]}
            if identifier == "ai4i":
                item["metrics_note"] = "沿用现有 CNC 权重，本次未重训或重新评测。"
            try:
                entry = self._cnc_entry(handler) if identifier == "ai4i" else self._entry(identifier)
                meta = entry["meta"]
                item.update(classes=meta["classes"], normal_class=meta["normal_class"],
                            feature_count=len(meta["raw_feature_names"]), display_features=meta["display_features"],
                            dataset=meta["dataset"])
                if identifier == "ai4i":
                    item["model_input_dim"] = meta["model_input_dim"]
                    model_info = self._cnc_model_info(handler)
                else:
                    item["model_input_dim"] = len(meta["selected_feature_indices"])
                    item["metrics"] = meta.get("metrics")
                    item["calibration"] = meta.get("calibration")
                    entry = self._model(identifier)
                    model_info = {"name": "industrial-mlp", "version": meta["model_version"],
                                  "device": entry["actual_device"], "sha256": entry["weight_hash"]}
                item.update(available=True, status="ready", model=model_info)
                if identifier == "steel":
                    item["limitations"].append("仅覆盖七类已标注缺陷，无正常类，不能据此计算正常/异常概率。")
                if identifier in ("skab", "secom"):
                    item["limitations"].append("二分类模型不提供具体故障类型或部件原因。")
                item["limitations"].extend(meta.get("limitations", []))
            except IndustrialError as error:
                item["unavailable_reason"] = str(error)
            scenarios.append(item)
        return {"scenarios": scenarios}

    def samples(self, scenario, handler, limit=24):
        entry = self._cnc_entry(handler) if scenario == "ai4i" else self._entry(scenario)
        # Interleave true classes so the first page includes examples of each available class.
        groups = defaultdict(list)
        for sample in entry["samples"]:
            groups[sample["expected_class"]].append(sample)
        selected = []
        for index in range(max(map(len, groups.values()))):
            for group in groups.values():
                if index < len(group) and len(selected) < limit:
                    selected.append(group[index])
            if len(selected) >= limit:
                break
        return {"scenario_id": scenario, "source": "heldout_dataset", "dataset": entry["meta"]["dataset"],
                "total": len(entry["samples"]), "samples": [
                    {**sample, "label": sample.get("label") or f"{sample['id']} · {sample['expected_class']}",
                     "display_features": display_values(entry["meta"]["display_features"], sample["features"])}
                    for sample in selected]}

    def decide(self, scenario, sample_id, handler):
        entry = self._cnc_entry(handler) if scenario == "ai4i" else self._entry(scenario)
        sample = entry["by_id"].get(sample_id)
        if sample is None:
            raise IndustrialError(404, "sample_not_found", "Sample does not belong to the selected scenario")
        if scenario == "ai4i":
            return self._decide_cnc(sample, entry, handler)
        entry = self._model(scenario)
        meta, torch, network = entry["meta"], entry["torch"], entry["network"]
        if entry["actual_device"].startswith("cuda"):
            torch.cuda.synchronize(network[0].weight.device)
        started = time.perf_counter()
        inputs = normalized_features(meta, sample["features"])
        with torch.inference_mode():
            tensor = torch.tensor([inputs], dtype=torch.float32, device=entry["actual_device"])
            probabilities = torch.softmax(network(tensor) / meta["temperature"], dim=1)[0].cpu().tolist()
        latency_ms = (time.perf_counter() - started) * 1000
        if not all(finite_number(p) and 0 <= p <= 1 for p in probabilities):
            raise IndustrialError(503, "invalid_model_output", "Model returned non-finite probabilities")
        distribution = dict(zip(meta["classes"], probabilities))
        label = max(distribution, key=distribution.get)
        normal = meta["normal_class"]
        is_anomaly = label != normal if normal is not None else None
        prediction = {"label": label, "probabilities": distribution, "is_anomaly": is_anomaly,
                      "fault_probability": 1 - distribution[normal] if normal is not None else None,
                      "anomaly_rule": "predicted_class_is_not_normal" if normal is not None else None,
                      "calibrated": meta["calibrated"],
                      "calibration_method": "temperature_scaling" if meta["calibrated"] else None}
        # This is an explicit review policy, not a learned severity or component diagnosis.
        policy = {"source": "derived_policy", "action": "observe" if is_anomaly is False else "human_review",
                  "reason": "模型类别为正常；保持观察。" if is_anomaly is False else "分类模型仅提供数据集类别判断，缺陷或异常交由人工复核。"}
        return {"scenario_id": scenario, "sample_id": sample_id, "source": "heldout_dataset", "mode": "live_model",
                "trace_id": str(uuid4()), "model": {"name": "industrial-mlp", "version": meta["model_version"],
                    "device": entry["actual_device"], "sha256": entry["weight_hash"]},
                "latency_ms": latency_ms, "prediction": prediction, "policy": policy,
                "sample": sample, "dataset": meta["dataset"],
                "display_features": display_values(meta["display_features"], sample["features"])}

    def _decide_cnc(self, sample, entry, handler):
        model_info = self._cnc_model_info(handler)
        payload = {"conversation_id": "industrial-ai4i-" + str(uuid4()), "current_message": "AI4I 留出测试样本回放",
                   "conversation_state": {"structured_features": sample["features"], "feature_units": CNC_UNITS},
                   "channel_metadata": {"source": "heldout_dataset", "scenario": "ai4i"}}
        started = time.perf_counter()
        status, result = handler.service.handle("/v1/decide", payload)
        latency_ms = (time.perf_counter() - started) * 1000
        if status != 200 or not isinstance(result, dict):
            raise IndustrialError(503, "cnc_inference_failed", f"Existing CNC service returned status {status}")
        decisions = {item.get("id"): item for item in result.get("decisions", []) if isinstance(item, dict)}
        fault, fault_type = decisions.get("fault", {}), decisions.get("fault_type", {})
        distribution = fault_type.get("probabilities", {})
        fault_probability = fault.get("probabilities", {}).get("yes")
        if (set(distribution) != set(CNC_CLASSES) or not all(finite_number(v) and 0 <= v <= 1 for v in distribution.values())
                or not math.isclose(sum(distribution.values()), 1, abs_tol=0.01)
                or not finite_number(fault_probability) or not 0 <= fault_probability <= 1
                or type(fault.get("value")) is not bool or fault_type.get("value") not in CNC_CLASSES
                or not isinstance(result.get("recommended_action"), dict) or not result.get("trace_id")):
            raise IndustrialError(503, "invalid_cnc_output", "Existing CNC service did not return valid fault and type predictions")
        providers = {fault.get("provider"), fault_type.get("provider")}
        if providers != {"sentinel-head"}:
            raise IndustrialError(503, "cnc_head_unavailable", "CNC response did not originate from the existing structured decision head")
        calibrated = all(item.get("calibration_status") in ("temperature_scaled", "distribution_shifted") for item in (fault, fault_type))
        return {**result, "scenario_id": "ai4i", "sample_id": sample["id"], "source": "heldout_dataset", "mode": "live_model",
                "model": model_info, "latency_ms": latency_ms, "sample": sample, "dataset": entry["meta"]["dataset"],
                "prediction": {"label": fault_type["value"], "probabilities": distribution, "is_anomaly": fault["value"],
                               "fault_probability": fault_probability, "anomaly_rule": "fault_head_probability_gte_0.5",
                               "calibrated": calibrated, "calibration_method": "temperature_scaling" if calibrated else None},
                "policy": {"source": "jev_runtime", "action": result["recommended_action"].get("kind"),
                           "reason": ", ".join(result["recommended_action"].get("reason_codes", []))},
                "display_features": display_values(CNC_FEATURES, sample["features"])}


_RUNTIMES = {}
_RUNTIME_LOCK = threading.Lock()


def runtime_for(handler, root):
    root = Path(root).resolve()
    device = handler.meta.get("device", "cpu")
    key = (str(root), str((root / "industrial-models-current").resolve()), device)
    with _RUNTIME_LOCK:
        if key not in _RUNTIMES:
            _RUNTIMES[key] = IndustrialRuntime(root, device)
        return _RUNTIMES[key]


def check_scenario(value):
    if not isinstance(value, str) or value not in SCENARIOS:
        raise IndustrialError(422, "invalid_scenario", "scenario must be ai4i, skab, steel, secom, or tep")
    return value


def send_error(handler, error):
    handler._json(error.status, {"error": str(error), "code": error.code, "mode": "unavailable"})


def handle_industrial_get(handler, root):
    url = urlsplit(handler.path)
    if url.path not in ("/v1/industrial/catalog", "/v1/industrial/samples"):
        return False
    try:
        runtime = runtime_for(handler, root)
        if url.path.endswith("/catalog"):
            result = runtime.catalog(handler)
        else:
            query = parse_qs(url.query)
            if len(query.get("scenario", [])) != 1:
                raise IndustrialError(422, "invalid_scenario", "Exactly one scenario is required")
            scenario = check_scenario(query["scenario"][0])
            try:
                if len(query.get("limit", ["24"])) != 1:
                    raise ValueError()
                limit = int(query.get("limit", ["24"])[0])
                if not 1 <= limit <= 100:
                    raise ValueError()
            except ValueError:
                raise IndustrialError(422, "invalid_limit", "limit must be an integer from 1 to 100")
            result = runtime.samples(scenario, handler, limit)
        handler._json(200, result)
    except IndustrialError as error:
        send_error(handler, error)
    except Exception:
        send_error(handler, IndustrialError(503, "industrial_unavailable", "Industrial runtime is unavailable"))
    return True


def handle_industrial_post(handler, root):
    if urlsplit(handler.path).path != "/v1/industrial/decide":
        return False
    try:
        try:
            length = int(handler.headers.get("content-length", "0"))
        except ValueError:
            raise IndustrialError(422, "invalid_body", "Content-Length must be an integer")
        if length > MAX_BODY_BYTES:
            raise IndustrialError(413, "body_too_large", "Request body exceeds 64 KiB")
        if length <= 0:
            raise IndustrialError(422, "invalid_body", "A JSON request body is required")
        try:
            payload = json.loads(handler.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeError):
            raise IndustrialError(422, "invalid_body", "Request body must contain valid JSON")
        if not isinstance(payload, dict):
            raise IndustrialError(422, "invalid_body", "Request body must be an object")
        scenario = check_scenario(payload.get("scenario_id"))
        sample_id = payload.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or len(sample_id) > 200:
            raise IndustrialError(422, "invalid_sample", "A valid sample_id is required")
        if set(payload) != {"scenario_id", "sample_id"}:
            raise IndustrialError(422, "invalid_body", "Only scenario_id and sample_id may be submitted")
        result = runtime_for(handler, root).decide(scenario, sample_id, handler)
        handler._json(200, result)
    except IndustrialError as error:
        send_error(handler, error)
    except Exception:
        send_error(handler, IndustrialError(503, "industrial_unavailable", "Industrial runtime is unavailable"))
    return True
