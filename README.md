# Dynamic AMI Risk Model Code

![Clinical scenario, dynamic risk, and clinical implications](Fig1.png)

Code release for landmark-level 24-hour mortality prediction in ICU patients with acute myocardial infarction.

## Structure

```text
src/dynamic_ami_risk/      package code
scripts/                   training and scoring
data/                      sample input
model/                     model files
tests/                     tests
```

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Run on the sample file

```bash
python scripts/predict_from_csv.py ^
  --input data/sample_landmark_features.csv ^
  --output outputs/predictions.csv ^
  --summary-json outputs/summary.json
```

## Train your own model

```bash
python scripts/train_from_private_csv.py ^
  --input your_training_data.csv ^
  --output-model model/lightgbm_model.txt ^
  --metadata-json model/model_metadata.json
```

## Score your own CSV

```bash
python scripts/predict_from_csv.py ^
  --input your_input.csv ^
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
