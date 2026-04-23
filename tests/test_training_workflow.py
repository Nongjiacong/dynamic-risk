from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dynamic_ami_risk.training import fit_lightgbm_model, prepare_training_frame, save_model_file


TRAIN_DATA_PATH = BASE_DIR / "data" / "sample_training_landmarks.csv"


def test_private_training_smoke(tmp_path):
    frame = pd.read_csv(TRAIN_DATA_PATH).head(500)
    features, labels = prepare_training_frame(frame)
    booster = fit_lightgbm_model(
        features,
        labels,
        num_boost_round=10,
        random_seed=20260418,
    )

    model_path = tmp_path / "smoke_model.txt"
    save_model_file(booster, model_path)

    assert model_path.exists()
