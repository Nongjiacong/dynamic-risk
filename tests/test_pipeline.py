from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dynamic_ami_risk import load_model, required_column_names, score_frame
from dynamic_ami_risk.settings import get_bundled_model_path


SAMPLE_DATA_PATH = BASE_DIR / "data" / "sample_landmark_features.csv"


def test_required_schema_is_stable():
    columns = required_column_names()
    assert len(columns) == 20
    assert "age" in columns
    assert "lactate_latest" in columns
    assert "vasoactive_support_current" in columns


def test_bundled_model_exists():
    assert get_bundled_model_path().exists()


def test_load_model_returns_configured_model():
    model = load_model()
    assert model.display_name
    assert model.alert_threshold > 0
    assert model.mode in {"file", "reference"}


def test_score_sample_dataset():
    frame = pd.read_csv(SAMPLE_DATA_PATH)
    result = score_frame(frame)

    assert result["summary"]["rows"] == len(frame)
    assert len(result["predictions"]) == len(frame)
    first_row = result["predictions"][0]
    assert "predicted_risk_24h" in first_row
    assert "alert" in first_row
    assert "risk_band" in first_row
