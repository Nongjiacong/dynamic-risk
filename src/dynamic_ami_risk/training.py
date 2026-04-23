from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Iterable

import lightgbm as lgb
import pandas as pd

from .feature_schema import required_column_names
from .settings import DEFAULT_PRODUCTION_METADATA


FEATURE_COLUMNS = required_column_names()
BINARY_COLUMNS = [
    "mechanical_ventilation_current",
    "vasoactive_support_current",
    "renal_replacement_therapy_current",
]
DEFAULT_LABEL_COLUMN = "observed_death_within_24h"
DEFAULT_RANDOM_SEED = 20260418


def to_binary(value: object) -> int:
    if pd.isna(value):
        return 0
    if isinstance(value, bool):
        return int(value)
    return 1 if str(value).strip().lower() in {"1", "true", "yes", "y", "male", "m"} else 0


def prepare_training_frame(
    frame: pd.DataFrame,
    label_column: str = DEFAULT_LABEL_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    if label_column not in frame.columns:
        raise ValueError(f"Training dataset must include {label_column}.")

    prepared = frame.copy()
    prepared["sex"] = prepared["sex"].map(to_binary).astype(int)

    for column in FEATURE_COLUMNS:
        if column == "sex":
            continue
        if column in BINARY_COLUMNS:
            prepared[column] = prepared[column].map(to_binary).astype(int)
        else:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    missing_after_coercion = [
        column for column in FEATURE_COLUMNS if prepared[column].isna().any()
    ]
    if missing_after_coercion:
        raise ValueError(
            "Some required training columns contain missing or non-numeric values after parsing: "
            + ", ".join(missing_after_coercion)
        )

    labels = prepared[label_column].map(to_binary).astype(int)
    return prepared[FEATURE_COLUMNS].astype(float), labels


def default_training_params(random_seed: int = DEFAULT_RANDOM_SEED) -> dict:
    return {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "learning_rate": 0.04,
        "num_leaves": 31,
        "min_data_in_leaf": 24,
        "feature_fraction": 0.90,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbose": -1,
        "seed": random_seed,
        "feature_fraction_seed": random_seed,
        "bagging_seed": random_seed,
    }


def fit_lightgbm_model(
    features: pd.DataFrame,
    labels: Iterable[int] | pd.Series,
    num_boost_round: int = 180,
    random_seed: int = DEFAULT_RANDOM_SEED,
    params: dict | None = None,
) -> lgb.Booster:
    train_set = lgb.Dataset(features, label=labels, feature_name=FEATURE_COLUMNS)
    resolved_params = params or default_training_params(random_seed=random_seed)
    return lgb.train(resolved_params, train_set, num_boost_round=num_boost_round)


def save_model_file(booster: lgb.Booster, model_path: Path) -> Path:
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temp_model_path = Path(tempfile.gettempdir()) / model_path.name
    booster.save_model(str(temp_model_path))
    shutil.copyfile(temp_model_path, model_path)
    return model_path


def build_production_metadata(
    display_name: str | None = None,
    version: str | None = None,
    alert_threshold: float | None = None,
    description: str | None = None,
    intended_use: str | None = None,
) -> dict:
    metadata = dict(DEFAULT_PRODUCTION_METADATA)
    if display_name is not None:
        metadata["display_name"] = display_name
    if version is not None:
        metadata["version"] = version
    if alert_threshold is not None:
        metadata["alert_threshold"] = alert_threshold
    if description is not None:
        metadata["description"] = description
    if intended_use is not None:
        metadata["intended_use"] = intended_use
    return metadata


def write_metadata(metadata_path: Path, metadata: dict) -> Path:
    metadata_path = Path(metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path
