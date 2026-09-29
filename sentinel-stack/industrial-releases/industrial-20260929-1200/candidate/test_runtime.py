from copy import deepcopy
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import industrial_runtime as runtime
import prepare_industrial_patch as prepare

try:
    import torch
except ImportError:
    torch = None


def metadata(scenario="skab", classes=None, normal="normal"):
    return {"scenario": scenario, "raw_feature_names": ["x", "y"],
            "classes": classes or ["normal", "fault"], "normal_class": normal,
            "model_version": "test-fixture-v1", "model_sha256": "0" * 64,
            "temperature": 2.0, "calibrated": True,
            "imputer_medians": [10.0, 20.0], "selected_feature_indices": [1],
            "scaler_mean": [10.0], "scaler_scale": [5.0],
            "dataset": {"name": "Unit-test fixture", "source": "synthetic fixture used only by tests"},
            "display_features": [{"key": "y", "label": "Recorded feature y", "unit": None, "kind": "model_feature"}]}


def samples(scenario="skab", classes=None):
    classes = classes or ["normal", "fault"]
    return [{"id": scenario + ":test:1", "label": "test sample", "split": "test",
             "features": {"x": 8.0, "y": None}, "expected_class": classes[-1]}]


class Handler:
    def __init__(self, path="/v1/industrial/decide", payload=None):
        self.path = path
        body = json.dumps(payload).encode() if payload is not None else b""
        self.headers = {"content-length": str(len(body)), "content-type": "application/json"}
        self.rfile = BytesIO(body)
        self.status, self.response = None, None
        self.engine = object()
        self.service = None
        self.meta = {"device": "cpu", "framework": "jev-decision-service 0.1.5", "decision_head": "sentinel-mlp@cpu"}

    def _json(self, status, response):
        self.status, self.response = status, response


class RuntimeTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.runtime = runtime.IndustrialRuntime(self.root, "cpu")
        runtime._RUNTIMES.clear()

    def write_artifacts(self, scenario="skab", meta=None, records=None):
        directory = self.root / "industrial-models-current" / scenario
        directory.mkdir(parents=True, exist_ok=True)
        meta = meta or metadata(scenario)
        records = records or samples(scenario, meta["classes"])
        (directory / "metadata.json").write_text(json.dumps(meta))
        (directory / "heldout_samples.json").write_text(json.dumps(records))
        return directory


class ContractTests(RuntimeTestCase):
    def test_missing_artifacts_are_503_not_simulated_success(self):
        handler = Handler(payload={"scenario_id": "skab", "sample_id": "skab:test:1"})
        self.assertTrue(runtime.handle_industrial_post(handler, self.root))
        self.assertEqual(handler.status, 503)
        self.assertEqual(handler.response["mode"], "unavailable")
        self.assertNotIn("prediction", handler.response)

    def test_original_routes_are_untouched(self):
        for path in ("/v1/decide", "/v1/decide/batch", "/healthz", "/dashboard", "/"):
            handler = Handler(path)
            self.assertFalse(runtime.handle_industrial_get(handler, self.root))
            self.assertFalse(runtime.handle_industrial_post(handler, self.root))
            self.assertIsNone(handler.status)

    def test_oversized_body_is_rejected_without_reading(self):
        handler = Handler()
        handler.headers["content-length"] = str(runtime.MAX_BODY_BYTES + 1)
        handler.rfile = SimpleNamespace(read=lambda *_: self.fail("Oversized body must not be read"))
        runtime.handle_industrial_post(handler, self.root)
        self.assertEqual(handler.status, 413)

    def test_invalid_payloads_are_422_before_model_access(self):
        for payload in ([], None, {}, {"scenario_id": "../secom", "sample_id": "x"},
                        {"scenario_id": "skab", "sample_id": 1},
                        {"scenario_id": "skab", "sample_id": "x", "features": {"y": 123}}):
            with self.subTest(payload=payload):
                handler = Handler(payload=payload)
                runtime.handle_industrial_post(handler, self.root)
                self.assertEqual(handler.status, 422)

    def test_invalid_json_and_content_length_are_422(self):
        for length, body in (("wrong", b"{}"), ("-1", b"{}"), ("1", b"{"), ("1", b"\xff")):
            handler = Handler()
            handler.headers["content-length"] = length
            handler.rfile = BytesIO(body)
            runtime.handle_industrial_post(handler, self.root)
            self.assertEqual(handler.status, 422)

    def test_duplicate_query_or_bad_limit_is_rejected(self):
        for query in ("", "scenario=skab&scenario=steel", "scenario=secom&limit=0", "scenario=tep&limit=101"):
            handler = Handler("/v1/industrial/samples?" + query)
            runtime.handle_industrial_get(handler, self.root)
            self.assertEqual(handler.status, 422)

    def test_sample_response_preserves_missing_values_and_source_inputs(self):
        self.write_artifacts()
        result = self.runtime.samples("skab", Handler())
        sample = result["samples"][0]
        self.assertEqual(sample["features"], {"x": 8.0, "y": None})
        self.assertIsNone(sample["display_features"][0]["value"])
        self.assertIsNone(sample["display_features"][0]["unit"])
        self.assertEqual(sample["split"], "test")
        self.assertEqual(result["source"], "heldout_dataset")

    def test_sample_id_cannot_silently_change_scenario(self):
        self.write_artifacts("skab")
        with self.assertRaises(runtime.IndustrialError) as raised:
            self.runtime.decide("skab", "steel:test:1", Handler())
        self.assertEqual(raised.exception.status, 404)

    def test_duplicate_training_and_bad_samples_are_rejected(self):
        good = samples()[0]
        bad_sets = [[good, good], [{**good, "split": "train"}],
                    [{**good, "features": {"x": 0}}],
                    [{**good, "features": {"x": math.nan, "y": None}}],
                    [{**good, "features": {"x": True, "y": None}}]]
        for records in bad_sets:
            with self.subTest(records=records):
                with self.assertRaises(ValueError):
                    runtime.validate_samples(records, ["x", "y"], ["normal", "fault"])

    def test_metadata_cannot_claim_another_scenario(self):
        self.write_artifacts("skab", meta=metadata("steel"))
        with self.assertRaisesRegex(runtime.IndustrialError, "scenario"):
            self.runtime._entry("skab")

    def test_preprocessing_uses_training_median_and_scaler(self):
        self.assertEqual(runtime.normalized_features(metadata(), {"x": 8.0, "y": None}), [2.0])
        self.assertEqual(runtime.normalized_features(metadata(), {"x": 8.0, "y": 15.0}), [1.0])

    def test_catalog_preserves_current_metrics_and_preprocessed_dimension(self):
        meta = metadata("secom")
        meta["metrics"] = {"n": 314, "balanced_accuracy": .4983, "macro_f1": .31,
                           "classification_rule": "argmax_calibrated_class_probability",
                           "noul": {"decision_rule": "fault_probability_greater_than_or_equal_to_threshold", "threshold": .5}}
        self.write_artifacts("secom", meta)
        entry = self.runtime._entry("secom")
        entry.update(actual_device="cpu", weight_hash=meta["model_sha256"])
        with patch.object(self.runtime, "_model", return_value=entry):
            catalog = self.runtime.catalog(Handler())
        secom = next(item for item in catalog["scenarios"] if item["id"] == "secom")
        self.assertEqual(secom["metrics"], meta["metrics"])
        self.assertEqual(secom["feature_count"], 2)
        self.assertEqual(secom["model_input_dim"], 1)
        cnc = next(item for item in catalog["scenarios"] if item["id"] == "ai4i")
        self.assertIsNone(cnc["metrics"])
        self.assertIn("未重训", cnc["metrics_note"])

    def test_invalid_preprocessing_metadata_is_rejected(self):
        for changes in ({"scaler_scale": [0]}, {"temperature": 0}, {"selected_feature_indices": [5]},
                        {"normal_class": "fake-normal"}, {"model_version": ""}, {"calibrated": "yes"}):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    runtime.validate_metadata({**metadata(), **changes})

    def cnc_fixture(self, result_status=200, provider="sentinel-head"):
        (self.root / "heads.pt").write_bytes(b"test fixture representing existing loaded weights")
        handler = Handler()
        result = {"trace_id": "existing-trace", "route": "fast", "recommended_action": {"kind": "answer_from_context", "reason_codes": ["existing-policy"]},
                  "decisions": [
                      {"id": "fault", "value": False, "probabilities": {"yes": .2, "no": .8}, "provider": provider, "calibration_status": "temperature_scaled"},
                      {"id": "fault_type", "value": "No Failure", "probabilities": dict(zip(runtime.CNC_CLASSES, [.75, .05, .05, .05, .05, .05])), "provider": provider, "calibration_status": "temperature_scaled"}]}
        called = []

        def handle(path, payload):
            called.append((path, deepcopy(payload)))
            return result_status, deepcopy(result)

        handler.service = SimpleNamespace(handle=handle)
        sample = {"id": "ai4i-test-1", "features": {item["key"]: 1.0 for item in runtime.CNC_FEATURES}, "split": "test", "expected_class": "No Failure"}
        entry = {"meta": {"dataset": {"name": "AI4I test fixture"}}, "samples": [sample], "by_id": {sample["id"]: sample}}
        self.runtime.entries["ai4i"] = entry
        return handler, sample, result, called

    def test_cnc_reuses_original_service_and_preserves_original_decision(self):
        handler, sample, original, called = self.cnc_fixture()
        result = self.runtime.decide("ai4i", sample["id"], handler)
        self.assertEqual(result["decisions"], original["decisions"])
        self.assertEqual(result["recommended_action"], original["recommended_action"])
        self.assertEqual(result["trace_id"], "existing-trace")
        self.assertEqual(result["prediction"]["fault_probability"], .2)
        self.assertEqual(result["policy"]["source"], "jev_runtime")
        path, payload = called[0]
        self.assertEqual(path, "/v1/decide")
        self.assertEqual(payload["conversation_state"]["structured_features"], sample["features"])
        self.assertEqual(payload["conversation_state"]["feature_units"], runtime.CNC_UNITS)
        self.assertNotIn("expected_class", json.dumps(payload))

    def test_cnc_failure_or_different_provider_is_not_fabricated(self):
        for status, provider in ((500, "sentinel-head"), (200, "rules"), (200, "stepfun")):
            handler, sample, _, _ = self.cnc_fixture(status, provider)
            with self.assertRaises(runtime.IndustrialError) as raised:
                self.runtime.decide("ai4i", sample["id"], handler)
            self.assertEqual(raised.exception.status, 503)


@unittest.skipUnless(torch is not None, "Run these real Torch tests in the existing Spark Python environment")
class TorchTests(RuntimeTestCase):
    def write_model(self, scenario="skab", classes=None, normal="normal"):
        meta = metadata(scenario, classes, normal)
        directory = self.write_artifacts(scenario, meta)
        network = torch.nn.Sequential(torch.nn.Linear(1, 64), torch.nn.ReLU(), torch.nn.Linear(64, 32), torch.nn.ReLU(), torch.nn.Linear(32, len(meta["classes"])))
        with torch.no_grad():
            for parameter in network.parameters():
                parameter.zero_()
            network[0].weight[0, 0] = 1
            network[2].weight[0, 0] = 1
            network[4].weight[0, 0] = -1
            network[4].weight[-1, 0] = 1
        torch.save(network.state_dict(), directory / "model.pt")
        meta["model_sha256"] = runtime.file_sha256(directory / "model.pt")
        (directory / "metadata.json").write_text(json.dumps(meta))
        return directory, meta

    def test_real_torch_inference_matches_preprocessed_input_and_temperature(self):
        _, meta = self.write_model()
        result = self.runtime.decide("skab", "skab:test:1", Handler())
        expected = math.exp(1) / (math.exp(-1) + math.exp(1))
        self.assertAlmostEqual(result["prediction"]["probabilities"]["fault"], expected, places=6)
        self.assertEqual(result["model"]["sha256"], meta["model_sha256"])
        self.assertEqual(result["model"]["device"], "cpu")
        self.assertTrue(result["prediction"]["is_anomaly"])
        self.assertEqual(result["policy"]["source"], "derived_policy")
        self.assertNotIn("severity", result["prediction"])
        self.assertIsNone(result["display_features"][0]["value"])

    def test_steel_never_invents_normal_class_probability_or_severity(self):
        classes = ["Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"]
        self.write_model("steel", classes, normal=None)
        result = self.runtime.decide("steel", "steel:test:1", Handler())
        self.assertEqual(set(result["prediction"]["probabilities"]), set(classes))
        self.assertIsNone(result["prediction"]["is_anomaly"])
        self.assertIsNone(result["prediction"]["fault_probability"])
        self.assertNotIn("severity", result)
        self.assertEqual(result["policy"]["action"], "human_review")

    def test_tampered_model_is_unavailable_before_prediction(self):
        directory, _ = self.write_model()
        (directory / "model.pt").write_bytes(b"tampered weights")
        with self.assertRaisesRegex(runtime.IndustrialError, "hash"):
            self.runtime.decide("skab", "skab:test:1", Handler())
        catalog = self.runtime.catalog(Handler())
        skab = next(item for item in catalog["scenarios"] if item["id"] == "skab")
        self.assertFalse(skab["available"])
        self.assertIsNone(skab["model"])


class PatchTests(unittest.TestCase):
    def test_patch_does_not_change_existing_inference_code(self):
        source = (prepare.IMPORT + "class Handler:\n" + prepare.GET + "        return 'existing GET'\n"
                  + prepare.POST + "        return 'existing POST'\n").encode()
        with patch.object(prepare, "EXPECTED_SHA256", hashlib.sha256(source).hexdigest()):
            result = prepare.patched_source(source)
        for added in (b"from industrial_runtime import handle_industrial_get, handle_industrial_post\n",
                      b"        if handle_industrial_get(self, ROOT):\n            return\n",
                      b"        if handle_industrial_post(self, ROOT):\n            return\n"):
            result = result.replace(added, b"", 1)
        self.assertEqual(result, source)

    def test_unreviewed_source_is_rejected(self):
        with self.assertRaises(ValueError):
            prepare.patched_source(b"different live service")


if __name__ == "__main__":
    unittest.main()
