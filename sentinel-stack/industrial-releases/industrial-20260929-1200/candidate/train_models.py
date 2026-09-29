#!/usr/bin/env python3
"""Train reproducible dataset heads; never synthesize inputs or reuse test labels for fitting."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import nn

SEED = 20260929
SCENARIOS = ("skab", "steel", "secom", "tep")
SKAB_NAMES = ["Accelerometer1RMS", "Accelerometer2RMS", "Current", "Pressure", "Temperature", "Thermocouple", "Voltage", "Volume Flow RateRMS"]
STEEL_NAMES = ["X_Minimum", "X_Maximum", "Y_Minimum", "Y_Maximum", "Pixels_Areas", "X_Perimeter", "Y_Perimeter", "Sum_of_Luminosity", "Minimum_of_Luminosity", "Maximum_of_Luminosity", "Length_of_Conveyer", "TypeOfSteel_A300", "TypeOfSteel_A400", "Steel_Plate_Thickness", "Edges_Index", "Empty_Index", "Square_Index", "Outside_X_Index", "Edges_X_Index", "Edges_Y_Index", "Outside_Global_Index", "LogOfAreas", "Log_X_Index", "Log_Y_Index", "Orientation_Index", "Luminosity_Index", "SigmoidOfAreas"]
STEEL_CLASSES = ["Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"]


@dataclass
class Dataset:
    key: str
    name: str
    X: np.ndarray
    y: np.ndarray
    names: list[str]
    classes: list[str]
    normal_class: str | None
    row_ids: list[str]
    splits: dict[str, np.ndarray]
    split_info: dict
    source_files: list[Path]
    display: list[dict]
    units: dict[str, str]


def chronological_split(n: int, gap: int = 0) -> dict[str, np.ndarray]:
    first, second = int(n * .6), int(n * .8)
    result = {"train": np.arange(first), "calibration": np.arange(first + gap, second), "test": np.arange(second + gap, n)}
    if any(not len(indices) for indices in result.values()):
        raise ValueError("insufficient rows for chronological split and gap")
    return result


def group_split(groups: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    unique = np.unique(groups)
    if len(unique) < 5:
        raise ValueError("at least five independent SKAB files are required")
    ordered = np.random.default_rng(seed).permutation(unique)
    a, b = int(len(ordered) * .6), int(len(ordered) * .8)
    return {name: np.flatnonzero(np.isin(groups, part)) for name, part in
            zip(("train", "calibration", "test"), (ordered[:a], ordered[a:b], ordered[b:]))}


def fit_preprocessor(raw_train: np.ndarray) -> dict:
    """Medians, selected columns, means and scales are fitted exclusively on training rows."""
    values = np.asarray(raw_train, dtype=np.float64)
    medians = np.array([np.median(col[np.isfinite(col)]) if np.isfinite(col).any() else 0.0 for col in values.T])
    filled = np.where(np.isfinite(values), values, medians)
    keep = np.flatnonzero(np.std(filled, axis=0) > 1e-12)
    if not len(keep):
        raise ValueError("training data has no nonconstant features")
    selected = filled[:, keep]
    return {"imputer_medians": medians.tolist(), "selected_feature_indices": keep.tolist(),
            "scaler_mean": selected.mean(0).tolist(), "scaler_scale": selected.std(0).tolist()}


def transform(raw: np.ndarray, preprocessing: dict) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(preprocessing["imputer_medians"]):
        raise ValueError("raw feature dimension mismatch")
    filled = np.where(np.isfinite(values), values, np.asarray(preprocessing["imputer_medians"]))
    selected = filled[:, preprocessing["selected_feature_indices"]]
    result = (selected - np.asarray(preprocessing["scaler_mean"])) / np.asarray(preprocessing["scaler_scale"])
    if not np.isfinite(result).all():
        raise ValueError("preprocessed values are not finite")
    return np.ascontiguousarray(result, dtype=np.float32)


def build_model(input_dim: int, class_count: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, class_count))


def train_model(X: np.ndarray, y: np.ndarray, class_count: int, epochs: int, device: str, seed: int):
    torch.manual_seed(seed)
    counts = np.bincount(y, minlength=class_count)
    if (counts == 0).any():
        raise ValueError("a declared class has no training examples")
    weights = len(y) / (class_count * counts)
    model = build_model(X.shape[1], class_count).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0001)
    loss_function = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    features = torch.tensor(X, dtype=torch.float32, device=device)
    labels = torch.tensor(y, dtype=torch.long, device=device)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(features), labels)
        if not torch.isfinite(loss):
            raise ValueError("training loss became nonfinite")
        loss.backward()
        optimizer.step()
    model.eval()
    return model, weights.tolist()


def logits_for(model, X: np.ndarray, device: str) -> np.ndarray:
    with torch.inference_mode():
        return model(torch.tensor(X, dtype=torch.float32, device=device)).cpu().numpy().astype(np.float64)


def probabilities(logits: np.ndarray, temperature: float) -> np.ndarray:
    scores = logits / temperature
    scores = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(scores)
    return exp / exp.sum(axis=1, keepdims=True)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> tuple[float, dict]:
    # This fixed grid uses calibration labels only; no model/epoch selection touches test data.
    candidates = np.unique(np.r_[np.geomspace(.05, 20, 161), 1.0])
    losses = [float(-np.log(np.clip(probabilities(logits, t)[np.arange(len(labels)), labels], 1e-12, 1)).mean()) for t in candidates]
    index = int(np.argmin(losses))
    base = float(-np.log(np.clip(probabilities(logits, 1)[np.arange(len(labels)), labels], 1e-12, 1)).mean())
    return float(candidates[index]), {"method": "scalar_temperature_grid", "fit_partition": "calibration", "n": len(labels), "nll_before": base, "nll_after": losses[index], "search_bounds": [.05, 20]}


def metrics(probs: np.ndarray, labels: np.ndarray, classes: list[str], normal_class: str | None) -> dict:
    prediction = probs.argmax(1)
    confidence = probs.max(1)
    ece = 0.0
    for i in range(10):
        selected = (confidence >= i / 10) & (confidence < (i + 1) / 10 if i < 9 else confidence <= 1)
        if selected.any():
            ece += selected.mean() * abs(confidence[selected].mean() - (prediction[selected] == labels[selected]).mean())
    result = {"n": len(labels), "classification_rule": "argmax_calibrated_class_probability", "accuracy": float(accuracy_score(labels, prediction)),
              "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
              "macro_f1": float(f1_score(labels, prediction, labels=np.arange(len(classes)), average="macro", zero_division=0)),
              "top_label_ece": float(ece), "multiclass_brier": float(np.mean(np.sum((probs - np.eye(len(classes))[labels]) ** 2, axis=1))),
              "confusion_matrix": confusion_matrix(labels, prediction, labels=np.arange(len(classes))).tolist(),
              "class_support": {label: int((labels == i).sum()) for i, label in enumerate(classes)}}
    if normal_class is not None:
        normal = classes.index(normal_class)
        fault_y = labels != normal
        fault_p = 1 - probs[:, normal]
        fault_prediction = fault_p >= .5
        positive = int(fault_y.sum())
        negative = int((~fault_y).sum())
        result["noul"] = {"decision_rule": "fault_probability_greater_than_or_equal_to_threshold", "threshold": .5,
                          "balanced_accuracy": float(balanced_accuracy_score(fault_y, fault_prediction)),
                          "recall": float(fault_prediction[fault_y].mean()) if positive else None,
                          "false_positive_rate": float(fault_prediction[~fault_y].mean()) if negative else None,
                          "brier": float(np.mean((fault_p - fault_y) ** 2)), "positive_support": positive, "negative_support": negative}
    return result


def display_fields(names: list[str], keys: list[str], physical=False):
    return [{"key": key, "label": key, "unit": None, "kind": "physical" if physical else "model_feature"} for key in keys if key in names]


def load_steel(root: Path, seed: int) -> Dataset:
    path = root / "x_steel/Faults.NNA"
    array = np.loadtxt(path)
    if array.ndim != 2 or array.shape[1] != 34 or not np.all(array[:, 27:].sum(1) == 1):
        raise ValueError("Steel requires 27 features and exactly one of seven labels")
    X, y = array[:, :27], array[:, 27:].argmax(1)
    indices = np.arange(len(y))
    train, rest = train_test_split(indices, test_size=.4, stratify=y, random_state=seed)
    calibration, test = train_test_split(rest, test_size=.5, stratify=y[rest], random_state=seed + 1)
    return Dataset("steel", "Steel Plates Faults", X, y, STEEL_NAMES, STEEL_CLASSES, None,
                   [f"Faults.NNA:{i + 1}" for i in indices], {"train": train, "calibration": calibration, "test": test},
                   {"strategy": "stratified_record_split_60_20_20", "limitation": "No plate/production-batch identifiers supplied; independent-batch generalization is unverified."}, [path],
                   display_fields(STEEL_NAMES, ["Pixels_Areas", "Steel_Plate_Thickness", "Luminosity_Index", "Edges_Index"]), {})


def load_secom(root: Path, seed: int) -> Dataset:
    path, label_path = root / "x_secom/secom.data", root / "x_secom/secom_labels.data"
    raw = np.genfromtxt(path, dtype=float)
    rows = [line.strip().split(maxsplit=1) for line in label_path.read_text().splitlines() if line.strip()]
    if raw.shape != (len(rows), 590):
        raise ValueError("SECOM must retain all 590 raw measurements before splitting")
    timestamps = [datetime.strptime(row[1].strip().strip('"'), "%d/%m/%Y %H:%M:%S") for row in rows]
    order = np.argsort(timestamps, kind="stable")
    y = np.array([int(float(row[0]) > 0) for row in rows])
    split = chronological_split(len(raw))
    names = [f"sensor_{i + 1:03d}" for i in range(590)]
    return Dataset("secom", "SECOM", raw[order], y[order], names, ["normal", "fault"], "normal",
                   [f"secom.data:{int(i) + 1}" for i in order], split,
                   {"strategy": "timestamp_order_60_20_20", "time_ranges": {key: [timestamps[int(order[ix[0]])].isoformat(), timestamps[int(order[ix[-1]])].isoformat()] for key, ix in split.items()},
                    "limitation": "Chronological holdout; unknown wafer/batch grouping and temporal drift remain limitations."}, [path, label_path],
                   display_fields(names, names[:4]), {})


def load_skab(root: Path, seed: int) -> Dataset:
    base = root / "x_skab/SKAB-master/data"
    arrays, labels, groups, row_ids, paths = [], [], [], [], []
    for path in sorted(base.rglob("*.csv")):
        with path.open(newline="", encoding="utf-8-sig") as source:
            first = source.readline()
            source.seek(0)
            reader = csv.DictReader(source, delimiter=";" if first.count(";") >= first.count(",") else ",")
            if not reader.fieldnames or "anomaly" not in reader.fieldnames or not set(SKAB_NAMES) <= set(reader.fieldnames):
                continue  # Unlabelled anomaly-free reference files cannot establish held-out labels.
            rows = list(reader)
        if not rows:
            continue
        X = np.array([[float(row[name]) for name in SKAB_NAMES] for row in rows])
        y = np.array([int(float(row["anomaly"])) for row in rows])
        if not np.isin(y, [0, 1]).all():
            raise ValueError(f"invalid SKAB anomaly label: {path.name}")
        group = str(path.relative_to(base))
        arrays.append(X); labels.append(y); groups.extend([group] * len(y))
        row_ids.extend([f"{group}:{i + 2}" for i in range(len(y))]); paths.append(path)
    if not arrays:
        raise ValueError("no labelled SKAB files found")
    grouping = np.array(groups)
    splits = group_split(grouping, seed)
    info = {"strategy": "whole_file_group_split_60_20_20", "groups": {key: sorted(set(grouping[ix])) for key, ix in splits.items()},
            "limitation": "Held-out experiment files, single SKAB testbed; no claim of cross-factory validation."}
    units = {"Accelerometer1RMS": "g", "Accelerometer2RMS": "g", "Current": "A", "Pressure": "bar", "Temperature": "°C", "Thermocouple": "°C", "Voltage": "V"}
    display = display_fields(SKAB_NAMES, ["Accelerometer1RMS", "Current", "Pressure", "Temperature"], physical=True)
    for field in display:
        field["unit"] = units[field["key"]]
    return Dataset("skab", "SKAB", np.vstack(arrays), np.concatenate(labels), SKAB_NAMES, ["normal", "fault"], "normal", row_ids, splits, info, paths, display, units)


def tep_matrix(path: Path) -> np.ndarray:
    values = np.loadtxt(path)
    if values.ndim != 2:
        raise ValueError(f"TEP array must be 2D: {path.name}")
    if values.shape[1] == 52:
        return values
    if values.shape[0] == 52:
        return values.T
    raise ValueError(f"TEP requires 52 process features: {path.name}")


def load_tep(root: Path, seed: int) -> Dataset:
    base = root / "x_tep/tennessee-eastman-profBraatz-master"
    arrays, targets, rows, paths = [], [], [], []
    splits = {"train": [], "calibration": [], "test": []}
    offset, gap = 0, 16
    for fault in range(22):
        for partition in ("tr", "te"):
            path = base / (f"d{fault:02d}.dat" if partition == "tr" else f"d{fault:02d}_te.dat")
            if not path.exists():
                raise ValueError(f"missing official TEP run {path}; refusing random-row fallback")
            values = tep_matrix(path)
            labels = np.full(len(values), fault, dtype=np.int64)
            if partition == "tr":
                boundary = int(.8 * len(values))
                splits["train"].extend(range(offset, offset + boundary))
                splits["calibration"].extend(range(offset + boundary + gap, offset + len(values)))
            else:
                # Exclude the initial pre-disturbance segment rather than label it a fault.
                start = 160 if fault > 0 else 0
                splits["test"].extend(range(offset + start, offset + len(values)))
            arrays.append(values); targets.append(labels); paths.append(path)
            rows.extend([f"{path.name}:{i + 1}" for i in range(len(values))]); offset += len(values)
    names = [f"XMEAS_{i}" for i in range(1, 42)] + [f"XMV_{i}" for i in range(1, 12)]
    info = {"strategy": "official_training_runs_80_20_chronological_calibration_gap16_official_test_runs_holdout",
            "calibration_gap_rows": gap, "fault_test_prefix_rows_excluded": 160,
            "limitation": "Calibration shares training-run trajectories after a gap; test uses separate official runs and excludes the first 160 rows of fault runs. Process simulation, not live plant telemetry."}
    display = [{"key": "XMEAS_7", "label": "反应器压力", "unit": "kPa gauge", "kind": "physical"},
               {"key": "XMEAS_9", "label": "反应器温度", "unit": "°C", "kind": "physical"},
               {"key": "XMEAS_2", "label": "D 进料流量", "unit": "kg/h", "kind": "physical"},
               {"key": "XMEAS_8", "label": "反应器液位", "unit": "%", "kind": "physical"}]
    return Dataset("tep", "Tennessee Eastman Process", np.vstack(arrays), np.concatenate(targets), names,
                   ["Normal"] + [f"Fault{i}" for i in range(1, 22)], "Normal", rows,
                   {key: np.array(indices, dtype=np.int64) for key, indices in splits.items()}, info, paths,
                   display, {field["key"]: field["unit"] for field in display})


LOADERS = {"steel": load_steel, "secom": load_secom, "skab": load_skab, "tep": load_tep}


def export_dataset(dataset: Dataset, output: Path, epochs: int, device: str, seed: int) -> dict:
    started = time.monotonic()
    parts = dataset.splits
    if any(not len(indices) for indices in parts.values()):
        raise ValueError("empty dataset partition")
    joined = np.concatenate(list(parts.values()))
    if len(np.unique(joined)) != len(joined):
        raise ValueError("train/calibration/test indices overlap")
    preprocessing = fit_preprocessor(dataset.X[parts["train"]])
    encoded = {key: transform(dataset.X[indices], preprocessing) for key, indices in parts.items()}
    model, weights = train_model(encoded["train"], dataset.y[parts["train"]], len(dataset.classes), epochs, device, seed)
    calibration_logits = logits_for(model, encoded["calibration"], device)
    temperature, calibration = fit_temperature(calibration_logits, dataset.y[parts["calibration"]])
    test_probs = probabilities(logits_for(model, encoded["test"], device), temperature)
    heldout_metrics = metrics(test_probs, dataset.y[parts["test"]], dataset.classes, dataset.normal_class)
    split_info = {**dataset.split_info, "counts": {key: len(indices) for key, indices in parts.items()},
                  "class_support": {key: {label: int((dataset.y[indices] == i).sum()) for i, label in enumerate(dataset.classes)} for key, indices in parts.items()},
                  "row_id_sha256": {key: hashlib.sha256("\n".join(dataset.row_ids[int(i)] for i in indices).encode()).hexdigest() for key, indices in parts.items()}}
    metadata = {"schema_version": 1, "scenario": dataset.key, "model_version": f"{dataset.key}-mlp64x32-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
                "created_at": datetime.now(timezone.utc).isoformat(), "model_architecture": {"hidden": [64, 32], "activation": "ReLU", "input_dim": len(preprocessing["selected_feature_indices"]), "output_dim": len(dataset.classes)},
                "raw_feature_names": dataset.names, "feature_units": dataset.units, "classes": dataset.classes, "normal_class": dataset.normal_class,
                "temperature": temperature, "calibrated": True, "calibration": calibration, **preprocessing,
                "dataset": {"name": dataset.name, "source": "local dataset files from ind_ds", "split": split_info,
                            "source_sha256": {str(path.relative_to(path.parents[0])) if len(dataset.source_files) == 1 else str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in dataset.source_files}},
                "training": {"seed": seed, "epochs": epochs, "optimizer": "Adam", "learning_rate": .001, "weight_decay": .0001, "class_weights": weights, "device": device,
                             "preprocessing_fit_partition": "train", "early_stopping": False},
                "metrics": heldout_metrics, "display_features": dataset.display,
                "limitations": ["Research dataset inference; not connected to physical sensors.", "Calibration fitted on a reserved partition; this is not production safety validation.", dataset.split_info["limitation"]]}
    samples = []
    # Include up to two real test records per class, never select on prediction correctness.
    for class_index, label in enumerate(dataset.classes):
        candidates = parts["test"][dataset.y[parts["test"]] == class_index]
        for row in candidates[:2]:
            features = {name: float(value) if np.isfinite(value) else None for name, value in zip(dataset.names, dataset.X[int(row)])}
            samples.append({"id": f"{dataset.key}:{dataset.row_ids[int(row)]}", "label": label, "expected_class": label,
                            "features": features, "split": "test", "source_row": dataset.row_ids[int(row)]})
    destination = output / dataset.key
    destination.mkdir(parents=True, exist_ok=False)
    torch.save({key: value.cpu() for key, value in model.state_dict().items()}, destination / "model.pt")
    metadata["model_sha256"] = hashlib.sha256((destination / "model.pt").read_bytes()).hexdigest()
    metadata["training"]["elapsed_seconds"] = round(time.monotonic() - started, 3)
    for name, value in [("metadata.json", metadata), ("heldout_samples.json", samples)]:
        (destination / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if not 1 <= args.epochs <= 2000:
        parser.error("epochs must be between 1 and 2000")
    torch.set_num_threads(4)
    for scenario in args.scenarios:
        if (args.output / scenario).exists():
            parser.error(f"refusing to overwrite existing artifact: {args.output / scenario}")
    for scenario in args.scenarios:
        dataset = LOADERS[scenario](args.data_root, args.seed)
        print(json.dumps({"scenario": scenario, "rows": len(dataset.y), "raw_features": len(dataset.names), "splits": {key: len(value) for key, value in dataset.splits.items()}}, ensure_ascii=False), flush=True)
        result = export_dataset(dataset, args.output, args.epochs, args.device, args.seed)
        print(json.dumps({"scenario": scenario, "artifact": str(args.output / scenario), "metrics": result["metrics"], "seconds": result["training"]["elapsed_seconds"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
