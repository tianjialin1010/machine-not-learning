"""Offline metrics from supplied predictions; this module does not train or call models."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from typing import Any


def evaluate_records(records: list[dict[str, Any]], bins: int = 10, allow_synthetic: bool = False) -> dict[str, Any]:
    if not records or not isinstance(bins, int) or bins < 1:
        raise ValueError("records must be nonempty and bins positive")
    required = {"sample_id", "dataset", "distribution", "truth", "probabilities", "provenance"}
    if any(not isinstance(r, dict) or not required <= set(r) for r in records):
        raise ValueError("missing evaluation fields")
    if any(r["provenance"] not in {"confirmed_label", "synthetic_fixture"} for r in records):
        raise ValueError("unknown label provenance")
    if any(r["provenance"] == "synthetic_fixture" for r in records) and not allow_synthetic:
        raise ValueError("synthetic fixtures require --allow-synthetic")
    if len({r["sample_id"] for r in records}) != len(records):
        raise ValueError("duplicate sample_id")
    if len({(r["dataset"], r["distribution"]) for r in records}) != 1:
        raise ValueError("evaluate one dataset/distribution group at a time")
    labels = sorted(records[0]["probabilities"])
    if len(labels) < 2 or any(not isinstance(x, str) for x in labels):
        raise ValueError("probabilities need at least two string labels")
    confusion = {truth: {pred: 0 for pred in labels} for truth in labels}
    buckets = [[] for _ in range(bins)]
    brier = 0.0
    timing = {"inference_ms": [], "e2e_ms": []}
    for row in records:
        probs = row["probabilities"]
        if set(probs) != set(labels) or row["truth"] not in labels:
            raise ValueError("inconsistent label space")
        if any(isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1 for p in probs.values()) or not math.isclose(sum(probs.values()),1,abs_tol=1e-6):
            raise ValueError("invalid probability distribution")
        selected = max(labels, key=lambda k: probs[k])
        confidence = probs[selected]
        correct = selected == row["truth"]
        confusion[row["truth"]][selected] += 1
        buckets[min(bins - 1, int(confidence * bins))].append((confidence, int(correct)))
        brier += sum((probs[label] - float(label == row["truth"])) ** 2 for label in labels)
        for key in timing:
            if row.get(key) is not None:
                value = row[key]
                if isinstance(value, bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value < 0:
                    raise ValueError("invalid latency")
                timing[key].append(value)
    n = len(records)
    per_class = {}
    for label in labels:
        tp = confusion[label][label]
        support = sum(confusion[label].values())
        predicted = sum(confusion[other][label] for other in labels)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else None
        f1 = 2 * tp / (support + predicted) if support + predicted else 0.0
        per_class[label] = {"support": support, "precision": precision, "recall": recall, "f1": f1}
    supported = all(v["support"] > 0 for v in per_class.values())
    ece = sum(len(b) / n * abs(sum(x[0] for x in b) / len(b) - sum(x[1] for x in b) / len(b)) for b in buckets if b)
    latency = {}
    for name, values in timing.items():
        values.sort()
        latency[name] = {"count": len(values), "p50": values[math.ceil(.50 * len(values))-1] if values else None, "p95": values[math.ceil(.95 * len(values))-1] if values else None}
    return {"dataset": records[0]["dataset"], "distribution": records[0]["distribution"], "count": n,
            "accuracy": sum(confusion[k][k] for k in labels) / n,
            "balanced_accuracy": sum(v["recall"] for v in per_class.values()) / len(labels) if supported else None,
            "macro_f1": sum(v["f1"] for v in per_class.values()) / len(labels),
            "brier_multiclass_sum": brier / n, "ece_top_label_equal_width": ece, "ece_bins": bins,
            "class_metrics": per_class, "confusion": confusion, "latency": latency,
            "warnings": [] if supported else ["missing_class_support"],
            "provenance": dict(Counter(r["provenance"] for r in records)),
            "scope": "metrics from supplied predictions only; model execution and source labels not independently verified"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSONL prediction records")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--allow-synthetic", action="store_true")
    args = parser.parse_args()
    groups = defaultdict(list)
    for line in args.input.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            groups[(row["dataset"], row["distribution"])].append(row)
    if not groups:
        parser.error("input has no records")
    results = [evaluate_records(rows, args.bins, args.allow_synthetic) for rows in groups.values()]
    print(json.dumps({"groups": results}, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
