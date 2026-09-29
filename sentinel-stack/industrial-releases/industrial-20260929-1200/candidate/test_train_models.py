"""CPU-only sanity checks on disposable synthetic arrays, never the deployment datasets."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from train_models import (Dataset, build_model, chronological_split, export_dataset,
                          fit_preprocessor, fit_temperature, group_split, load_secom, load_steel,
                          load_tep, metrics, probabilities, transform)


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_imputation_and_column_selection_use_training_only(self):
        train = np.array([[1, np.nan, 9, np.nan], [3, 10, 9, np.nan], [5, 20, 9, np.nan]])
        fitted = fit_preprocessor(train)
        self.assertEqual(fitted["imputer_medians"], [3, 15, 9, 0])
        self.assertEqual(fitted["selected_feature_indices"], [0, 1])
        snapshot = json.dumps(fitted, sort_keys=True)
        encoded = transform(np.array([[999, np.nan, -100, 500]]), fitted)
        self.assertEqual(encoded.shape, (1, 2))
        self.assertAlmostEqual(float(encoded[0, 1]), 0)
        self.assertEqual(json.dumps(fitted, sort_keys=True), snapshot)
        self.assertTrue(np.isfinite(encoded).all())
        with self.assertRaises(ValueError):
            transform(np.zeros((1, 3)), fitted)

    def test_constant_only_training_data_is_rejected(self):
        with self.assertRaises(ValueError):
            fit_preprocessor(np.ones((10, 3)))

    def test_group_split_never_splits_a_file_or_reuses_rows(self):
        groups = np.repeat([f"run-{i}" for i in range(15)], 20)
        parts = group_split(groups, 19)
        again = group_split(groups, 19)
        assigned = {}
        for split, indices in parts.items():
            np.testing.assert_array_equal(indices, again[split])
            for group in set(groups[indices]):
                self.assertNotIn(group, assigned)
                assigned[group] = split
                self.assertEqual(int((groups[indices] == group).sum()), 20)
        self.assertEqual(len(assigned), 15)
        self.assertEqual(len(np.unique(np.concatenate(list(parts.values())))), len(groups))

    def test_chronological_gap_separates_adjacent_partitions(self):
        split = chronological_split(100, gap=4)
        self.assertEqual(split["train"][-1], 59)
        self.assertEqual(split["calibration"][0], 64)
        self.assertEqual(split["calibration"][-1], 79)
        self.assertEqual(split["test"][0], 84)

    def test_temperature_fit_does_not_worsen_calibration_nll(self):
        logits = np.array([[8, 0], [0, 8], [0, 8], [8, 0]], dtype=float)
        labels = np.array([0, 1, 0, 1])
        temperature, report = fit_temperature(logits, labels)
        self.assertGreater(temperature, 1)
        self.assertLessEqual(report["nll_after"], report["nll_before"])
        np.testing.assert_allclose(probabilities(logits, temperature).sum(1), 1)

    def test_multiclass_argmax_and_binary_fault_threshold_are_distinct_named_metrics(self):
        values = np.array([[.4, .3, .3], [.8, .1, .1], [.1, .8, .1]])
        report = metrics(values, np.array([1, 0, 1]), ["Normal", "Fault1", "Fault2"], "Normal")
        self.assertEqual(report["classification_rule"], "argmax_calibrated_class_probability")
        self.assertEqual(report["noul"]["threshold"], .5)
        self.assertEqual(report["noul"]["decision_rule"], "fault_probability_greater_than_or_equal_to_threshold")
        self.assertAlmostEqual(report["accuracy"], 2 / 3)
        self.assertEqual(report["noul"]["recall"], 1)

    def test_steel_never_includes_one_hot_labels_as_features(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "x_steel").mkdir()
            rng = np.random.default_rng(5)
            features = rng.normal(size=(140, 27))
            labels = np.tile(np.arange(7), 20)
            np.savetxt(root / "x_steel/Faults.NNA", np.c_[features, np.eye(7)[labels]])
            dataset = load_steel(root, 42)
            np.testing.assert_array_equal(dataset.X, features)
            self.assertEqual(dataset.X.shape[1], 27)
            self.assertIsNone(dataset.normal_class)
            self.assertEqual(len(dataset.classes), 7)
            for indices in dataset.splits.values():
                self.assertEqual(len(set(dataset.y[indices])), 7)

    def test_secom_quoted_timestamps_are_parsed_before_chronological_split(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base = root / "x_secom"
            base.mkdir()
            raw = np.repeat(np.arange(20)[:, None], 590, axis=1).astype(float)
            raw[0, 7] = np.nan
            np.savetxt(base / "secom.data", raw)
            labels = [f'{1 if i % 2 else -1} "{20-i:02d}/07/2008 11:55:00"' for i in range(20)]
            (base / "secom_labels.data").write_text("\n".join(labels) + "\n")
            dataset = load_secom(root, 42)
            np.testing.assert_array_equal(dataset.X[:, 0], np.arange(19, -1, -1))
            self.assertEqual(dataset.X.shape[1], 590)
            self.assertTrue(np.isnan(dataset.X[-1, 7]))
            self.assertEqual(dataset.row_ids[0], "secom.data:20")
            self.assertLess(dataset.split_info["time_ranges"]["train"][1], dataset.split_info["time_ranges"]["calibration"][0])
            self.assertLess(dataset.split_info["time_ranges"]["calibration"][1], dataset.split_info["time_ranges"]["test"][0])

    def test_tep_official_test_files_are_not_training_and_prefix_is_excluded(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base = root / "x_tep/tennessee-eastman-profBraatz-master"
            base.mkdir(parents=True)
            for fault in range(22):
                train = np.full((100, 52), fault, dtype=float)
                test = np.full((180, 52), fault + 100, dtype=float)
                np.savetxt(base / f"d{fault:02d}.dat", train)
                np.savetxt(base / f"d{fault:02d}_te.dat", test)
            dataset = load_tep(root, 42)
            for partition in ("train", "calibration"):
                self.assertTrue(all("_te.dat" not in dataset.row_ids[i] for i in dataset.splits[partition]))
            for index in dataset.splits["test"]:
                row = dataset.row_ids[index]
                self.assertIn("_te.dat", row)
                if not row.startswith("d00_"):
                    self.assertGreater(int(row.split(":")[-1]), 160)
            self.assertEqual(len(dataset.classes), 22)
            self.assertEqual(dataset.X.shape[1], 52)
            self.assertEqual(len(set(dataset.y[dataset.splits["train"]])), 22)

    def test_cpu_training_exports_reloadable_model_and_only_heldout_raw_samples(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "synthetic-fixture.txt"
            source.write_text("CPU test fixture; not a production training artifact")
            rng = np.random.default_rng(17)
            features = rng.normal(size=(90, 3))
            features[:, 2] = 3
            features[2, 1] = np.nan
            labels = np.tile([0, 1], 45)
            dataset = Dataset("skab", "CPU synthetic sanity", features, labels, ["a", "b", "constant"], ["normal", "fault"], "normal",
                              [f"fixture:{i}" for i in range(90)], chronological_split(90),
                              {"strategy": "synthetic-test-only", "limitation": "Synthetic unit test"}, [source], [], {})
            metadata = export_dataset(dataset, root / "artifacts", epochs=4, device="cpu", seed=12)
            artifact = root / "artifacts/skab"
            self.assertEqual(metadata["training"]["device"], "cpu")
            self.assertEqual(metadata["training"]["preprocessing_fit_partition"], "train")
            self.assertEqual(metadata["model_sha256"], hashlib.sha256((artifact / "model.pt").read_bytes()).hexdigest())
            self.assertEqual(metadata["selected_feature_indices"], [0, 1])
            model = build_model(2, 2)
            state = torch.load(artifact / "model.pt", map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=True)
            samples = json.loads((artifact / "heldout_samples.json").read_text())
            heldout = {dataset.row_ids[i] for i in dataset.splits["test"]}
            training = {dataset.row_ids[i] for i in dataset.splits["train"]}
            self.assertTrue(samples)
            self.assertEqual(len({item["id"] for item in samples}), len(samples))
            for sample in samples:
                self.assertEqual(sample["split"], "test")
                self.assertIn(sample["source_row"], heldout)
                self.assertNotIn(sample["source_row"], training)
                self.assertEqual(set(sample["features"]), set(dataset.names))
                raw = np.array([[sample["features"][name] for name in dataset.names]], dtype=float)
                with torch.inference_mode():
                    prediction = model(torch.tensor(transform(raw, metadata)))
                self.assertTrue(torch.isfinite(prediction).all())
            with self.assertRaises(FileExistsError):
                export_dataset(dataset, root / "artifacts", epochs=1, device="cpu", seed=12)


if __name__ == "__main__":
    unittest.main()
