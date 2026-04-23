# Dynamic AMI Risk Model Code

Code release for landmark-level 24-hour mortality prediction in ICU patients with acute myocardial infarction.

## Structure

```text
src/dynamic_ami_risk/      package code
scripts/                   data generation, training, scoring
data/                      public sample files
model/                     model files
tests/                     tests
```

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Generate public sample files

```bash
python scripts/generate_sample_data.py
```

## Train bundled model

```bash
python scripts/train_bundled_model.py
```

## Train your own model

```bash
python scripts/train_from_private_csv.py ^
  --input your_training_data.csv ^
  --output-model model/lightgbm_model.txt ^
  --metadata-json model/model_metadata.json
```

## Score a CSV

```bash
python scripts/predict_from_csv.py ^
  --input data/sample_landmark_features.csv ^
  --output outputs/predictions.csv ^
  --summary-json outputs/summary.json
```

## Model loading order

1. `model/lightgbm_model.txt`
2. `model/bundled_lightgbm_model.txt`
3. built-in fallback scorer

## Quick run on Windows

```bash
examples\run_sample.bat
```

## Notes

- Public files in `data/` are synthetic.
- Restricted source data are not included.
- The repository expects prepared landmark-level feature tables, not raw EHR tables.
