# Model Card

## Model Name

Dynamic AMI 24-hour Mortality Risk Model

## Purpose

This repository exposes a landmark-level prediction model intended to estimate the risk of in-hospital death within the next 24 hours for critically ill patients with acute myocardial infarction in ICU settings.

## Intended Use

- Research-oriented scoring of prepared landmark-level feature tables
- Internal benchmarking of exported LightGBM model files against a fixed feature schema

## Not Intended For

- Direct autonomous clinical decision-making
- Use on raw EHR tables without validated preprocessing and harmonization
- Use outside the landmark-based ICU AMI setting without re-evaluation

## Expected Input

The model expects one row per landmark observation with harmonized feature columns documented in the repository README.

Examples include:

- baseline variables such as `age` and `sex`
- recent physiologic variables such as `heart_rate_latest`, `map_latest`, and `spo2_latest`
- recent laboratory variables such as `lactate_latest`, `creatinine_latest`, and `glucose_latest`
- support variables such as `mechanical_ventilation_current` and `vasoactive_support_current`

## Output

- `predicted_risk_24h`: predicted probability of death within the next 24 hours
- `alert`: threshold-based flag
- `risk_band`: simple low / intermediate / high grouping

## Threshold

The packaged code uses a default alert threshold of `0.10` unless overridden by model metadata or the `ALERT_THRESHOLD` environment variable.

## Validation Context

The published study design was based on a landmarking framework in ICU patients with AMI, with model development in eICU-CRD and external validation in MIMIC-IV.

This public repository does not reproduce the full restricted-data training workflow. It focuses on model implementation and scoring.

## Limitations

- Performance depends on correct feature engineering before scoring
- Missing or improperly harmonized units may invalidate predictions
- Predictions should be interpreted in the context of repeated landmark updates rather than as one-time admission risk
- The bundled LightGBM model file is included for repository use and is not the manuscript model

## Governance Recommendation

Before scientific distribution or controlled deployment, document:

- model version
- exact input schema
- threshold selection rationale
- calibration monitoring plan
- update and rollback procedure
