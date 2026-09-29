# Industrial dataset model export

`train_models.py` creates new, reproducible SKAB, Steel Plates, SECOM and TEP model artifacts. It does not reproduce the older dashboard's historical metrics, fabricate sensor readings, or connect to physical equipment. The only dependencies are the existing NumPy, PyTorch and scikit-learn installations.

Run the CPU sanity checks before using the real datasets:

```sh
cd demo/industrial
python -m unittest -v test_train_models
```

Train into a **new** output directory on the Spark:

```sh
python train_models.py \
  --data-root /home/USER/jev-service/ind_ds \
  --output /path/to/new-release/artifacts \
  --epochs 300 --device cuda --seed 20260929
```

The script refuses to overwrite a scenario artifact directory. It neither starts a server nor alters existing deployed weights. `--scenarios skab steel secom tep` can select a subset. An unavailable dataset or missing training class causes a failure; there is no synthetic-data fallback.

## Data separation

| Dataset | Raw inputs and targets | Split |
|---|---|---|
| SKAB | Eight sensor columns; only `anomaly` is the target. `datetime` and `changepoint` are excluded from features. | Seeded 60/20/20 split of whole experiment CSV files. A file never contributes rows to multiple partitions. |
| Steel Plates | First 27 columns; the final seven one-hot columns are labels. Seven defect classes, no normal class. | Stratified 60/20/20 record split. No plate or production-batch IDs are available, so independent-batch generalization is unverified. |
| SECOM | All 590 original measurements, including missing values; pass/fail comes from the separate labels file. | Timestamp-ordered 60/20/20 split. Original column positions are preserved through named `sensor_001` … `sensor_590` inputs. |
| TEP | `XMEAS(1..41)` followed by `XMV(1..11)`; Normal plus Fault1 … Fault21. | Official `d00.dat` … `d21.dat` training runs: first 80% for training, then a 16-row gap, remainder for calibration. Official `d00_te.dat` … `d21_te.dat` are test only. The first 160 rows of fault test runs are excluded rather than mislabelled as faults. |

SECOM imputation medians, all datasets' constant-column selection, scaling means and scaling standard deviations are **fitted only on training rows**. An entirely missing training column is filled with zero and removed as constant. Neither calibration nor test data influences preprocessing. The number of retained SECOM dimensions is measured from the actual training subset, not forced to the historical value 474.

The MLP is `Linear(input,64) → ReLU → Linear(64,32) → ReLU → Linear(32,classes)`. Training uses a fixed epoch count and inverse-frequency weights computed from training labels. There is no test-based early stopping or model selection. A single temperature is selected by calibration-set negative log likelihood. Independent test metrics include class support, confusion matrix, accuracy, balanced accuracy, macro F1, top-label ECE and Brier score. Binary anomaly metrics are additionally calculated when the dataset has a normal class.

`calibrated: true` means temperature scaling was fitted on the designated calibration partition. It does not claim production safety validation or guarantee accurate out-of-distribution predictions. Test accuracy and all limitations must remain visible alongside that status.

## Artifact contract

Each scenario writes:

```text
artifacts/<scenario>/
  model.pt
  metadata.json
  heldout_samples.json
```

- `model.pt` contains only the PyTorch `state_dict`. Parameter keys are `0.weight`, `0.bias`, `2.weight`, `2.bias`, `4.weight`, `4.bias`; the runtime must load strictly and use `weights_only=True`.
- `metadata.json` contains `model_version`, `model_sha256`, `model_architecture`, `raw_feature_names`, `imputer_medians`, `selected_feature_indices`, `scaler_mean`, `scaler_scale`, `classes`, `normal_class`, `temperature`, `calibrated`, `calibration`, `dataset`, `training`, `metrics`, `display_features` and `limitations`.
- `dataset.split` records the split strategy, counts, per-class support and ordered source-row-ID hashes. SKAB additionally records each partition's file list; SECOM records time ranges. Source-file SHA-256 hashes identify the exact dataset files used.
- `heldout_samples.json` contains up to two **test** records per true class, chosen in source order without filtering for correct predictions. Each record contains `id`, `features`, `expected_class`, `split: "test"` and `source_row`. Missing measurements are JSON `null`; labels never appear inside `features`.

Inference reconstructs raw feature order, replaces missing values with saved medians, selects saved columns, applies saved scaling, runs the network, and computes `softmax(logits / temperature)`. For SKAB, SECOM and TEP, anomaly probability is `1 - P(normal_class)`. Steel is classification-only and must not display an invented normal/fault probability.

## Units and limitations

SKAB sensor units are taken from the dataset README: acceleration in g, current in A, pressure in bar, temperatures in °C and voltage in V. Its historical flow-column spelling differs from the README's `RateRMS`, so no flow unit is asserted here. TEP's four displayed variables and units come from its official README: reactor pressure, reactor temperature, D-feed flow and reactor level. Steel and SECOM feature displays are labelled model features with unknown units rather than fabricated physical sensors.

The source layouts were checked against the Spark copies of `multi_dataset_bench.py`, TEP `readme.txt`/`README.md`, and SKAB `README.md`. This exporter corrects the old script's all-data SECOM preprocessing and random row splits of temporal datasets. TEP is process-simulation data; SKAB is one experimental testbed. These limitations do not disappear when the models run on a real GPU.
