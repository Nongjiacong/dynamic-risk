from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dynamic_ami_risk.training import fit_lightgbm_model, prepare_training_frame, save_model_file


def _training_frame() -> pd.DataFrame:
    rows = []
    for idx in range(24):
        rows.append(
            {
                "age": 62 + (idx % 8),
                "sex": "male" if idx % 2 == 0 else "female",
                "heart_rate_latest": 82 + idx,
                "map_latest": 78 - (idx % 6),
                "respiratory_rate_latest": 18 + (idx % 5),
                "spo2_latest": 97 - (idx % 3),
                "gcs_total_latest": 15 - (idx % 4 == 0),
                "urine_output_24h": 1700 - idx * 35,
                "lactate_latest": 1.4 + idx * 0.08,
                "creatinine_latest": 0.9 + idx * 0.03,
                "bun_latest": 16 + idx,
                "wbc_latest": 8.5 + idx * 0.2,
                "hemoglobin_latest": 13.4 - idx * 0.08,
                "platelet_latest": 240 - idx * 3,
                "glucose_latest": 120 + idx * 2,
                "mechanical_ventilation_current": int(idx % 5 == 0),
                "vasoactive_support_current": int(idx % 6 == 0),
                "renal_replacement_therapy_current": int(idx % 11 == 0),
                "measurement_count_24h": 6 + (idx % 4),
                "hours_since_last_measurement": 1.0 + (idx % 5) * 0.5,
                "observed_death_within_24h": int(idx % 4 == 0),
            }
        )
    return pd.DataFrame(rows)


def test_private_training_smoke(tmp_path):
    frame = _training_frame()
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
