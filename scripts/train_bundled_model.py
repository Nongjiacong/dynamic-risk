from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dynamic_ami_risk.settings import get_bundled_model_path
from dynamic_ami_risk.training import fit_lightgbm_model, prepare_training_frame, save_model_file
from scripts.generate_sample_data import TRAIN_OUTPUT_PATH, generate_sample_datasets


LABEL_COLUMN = "observed_death_within_24h"
RANDOM_SEED = 20260418


def _load_training_frame() -> pd.DataFrame:
    if not TRAIN_OUTPUT_PATH.exists():
        generate_sample_datasets()
    return pd.read_csv(TRAIN_OUTPUT_PATH)


def train_bundled_model() -> Path:
    training_frame = _load_training_frame()
    features, labels = prepare_training_frame(training_frame, label_column=LABEL_COLUMN)
    booster = fit_lightgbm_model(
        features,
        labels,
        num_boost_round=180,
        random_seed=RANDOM_SEED,
    )

    model_path = get_bundled_model_path()
    save_model_file(booster, model_path)

    print(f"Saved bundled model file to {model_path}")
    print(f"Training rows: {len(training_frame)}")
    print(f"Positive rate: {float(labels.mean()):.4f}")
    return model_path


if __name__ == "__main__":
    train_bundled_model()
