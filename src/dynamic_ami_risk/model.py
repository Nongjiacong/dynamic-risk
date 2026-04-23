from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .feature_schema import OPTIONAL_COLUMNS, required_column_names
from .settings import (
    get_bundled_model_path,
    get_production_model_path,
    load_model_metadata,
)

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None


FEATURE_COLUMNS = required_column_names()
BINARY_COLUMNS = [
    "mechanical_ventilation_current",
    "vasoactive_support_current",
    "renal_replacement_therapy_current",
]
PASSTHROUGH_COLUMNS = OPTIONAL_COLUMNS


def _to_binary(value: object) -> int:
    if pd.isna(value):
        return 0
    if isinstance(value, bool):
        return int(value)
    text = str(value).strip().lower()
    return 1 if text in {"1", "true", "yes", "y", "male", "m"} else 0


def _prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    prepared["sex"] = prepared["sex"].map(_to_binary)

    for column in FEATURE_COLUMNS:
        if column == "sex":
            continue
        if column in BINARY_COLUMNS:
            prepared[column] = prepared[column].map(_to_binary)
        else:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    missing_after_coercion = [
        column for column in FEATURE_COLUMNS if prepared[column].isna().any()
    ]
    if missing_after_coercion:
        raise ValueError(
            "Some required columns contain missing or non-numeric values after parsing: "
            + ", ".join(missing_after_coercion)
        )

    return prepared[FEATURE_COLUMNS].astype(float)


class ReferenceDynamicAmiModel:
    mode = "reference"
    model_source = "reference"
    model_path = None

    def __init__(self, metadata: dict) -> None:
        self.metadata = metadata
        self.name = metadata["short_name"]
        self.display_name = metadata["display_name"]
        self.alert_threshold = float(metadata["alert_threshold"])

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        features = _prepare_features(frame)

        logit = (
            -4.75
            + 0.028 * np.maximum(features["age"] - 65.0, 0.0)
            + 0.020 * np.maximum(features["heart_rate_latest"] - 85.0, 0.0)
            + 0.045 * np.maximum(70.0 - features["map_latest"], 0.0)
            + 0.030 * np.maximum(features["respiratory_rate_latest"] - 20.0, 0.0)
            + 0.080 * np.maximum(95.0 - features["spo2_latest"], 0.0)
            + 0.220 * np.maximum(15.0 - features["gcs_total_latest"], 0.0)
            + 0.004 * np.maximum(1500.0 - features["urine_output_24h"], 0.0) / 100.0
            + 0.420 * np.maximum(features["lactate_latest"] - 2.0, 0.0)
            + 0.120 * np.maximum(features["creatinine_latest"] - 1.2, 0.0)
            + 0.040 * np.maximum(features["bun_latest"] - 20.0, 0.0)
            + 0.020 * np.maximum(features["wbc_latest"] - 11.0, 0.0)
            + 0.060 * np.maximum(12.0 - features["hemoglobin_latest"], 0.0)
            + 0.002 * np.maximum(200.0 - features["platelet_latest"], 0.0)
            + 0.008 * np.maximum(features["glucose_latest"] - 160.0, 0.0)
            + 0.060 * features["sex"]
            + 0.550 * features["mechanical_ventilation_current"]
            + 0.750 * features["vasoactive_support_current"]
            + 0.650 * features["renal_replacement_therapy_current"]
            + 0.015 * np.maximum(features["measurement_count_24h"] - 6.0, 0.0)
            + 0.030 * np.maximum(features["hours_since_last_measurement"] - 4.0, 0.0)
        )

        probability = 1.0 / (1.0 + np.exp(-logit))
        return pd.Series(probability, index=frame.index, name="predicted_risk_24h")


class LightGBMFileModel:
    mode = "file"

    def __init__(self, model_path: Path, model_source: str, metadata: dict) -> None:
        if lgb is None:  # pragma: no cover
            raise RuntimeError("lightgbm is not installed.")
        self.metadata = metadata
        self.name = metadata["short_name"]
        self.display_name = metadata["display_name"]
        self.alert_threshold = float(metadata["alert_threshold"])
        self.model_path = Path(model_path)
        self.model_source = model_source
        self.booster = lgb.Booster(model_str=self.model_path.read_text(encoding="utf-8"))

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        features = _prepare_features(frame)
        prediction = self.booster.predict(features[FEATURE_COLUMNS])
        return pd.Series(prediction, index=frame.index, name="predicted_risk_24h")


def _load_file_model(model_path: Path, profile: str):
    if not model_path.exists() or lgb is None:
        return None
    metadata = load_model_metadata(profile)
    return LightGBMFileModel(model_path, model_source=profile, metadata=metadata)


def load_model():
    production_model = _load_file_model(
        get_production_model_path(),
        profile="production",
    )
    if production_model is not None:
        return production_model

    bundled_model = _load_file_model(
        get_bundled_model_path(),
        profile="bundled",
    )
    if bundled_model is not None:
        return bundled_model

    return ReferenceDynamicAmiModel(load_model_metadata("reference"))


def risk_band(probability: float) -> str:
    if probability < 0.03:
        return "low"
    if probability < 0.10:
        return "intermediate"
    return "high"


def score_frame(frame: pd.DataFrame, model=None) -> Dict[str, object]:
    active_model = model or load_model()
    probabilities = active_model.predict(frame)
    output = frame.copy()
    output["predicted_risk_24h"] = probabilities.round(4)
    output["alert"] = output["predicted_risk_24h"] >= active_model.alert_threshold
    output["risk_band"] = output["predicted_risk_24h"].map(risk_band)

    summary = {
        "rows": int(len(output)),
        "mean_risk": float(output["predicted_risk_24h"].mean()),
        "median_risk": float(output["predicted_risk_24h"].median()),
        "max_risk": float(output["predicted_risk_24h"].max()),
        "alert_count": int(output["alert"].sum()),
        "alert_threshold": float(active_model.alert_threshold),
    }

    return {
        "model": {
            "name": active_model.name,
            "display_name": active_model.display_name,
            "mode": active_model.mode,
            "model_loaded": active_model.mode == "file",
            "model_source": getattr(active_model, "model_source", "reference"),
            "model_path": (
                str(active_model.model_path)
                if getattr(active_model, "model_path", None) is not None
                else None
            ),
            "description": active_model.metadata.get("description", ""),
            "version": active_model.metadata.get("version", "0.1.0"),
            "intended_use": active_model.metadata.get("intended_use", ""),
        },
        "summary": summary,
        "predictions": output.to_dict(orient="records"),
    }
