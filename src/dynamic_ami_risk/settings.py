from __future__ import annotations

import json
import os
from pathlib import Path


BASE_DIR = Path(__file__).parent.parent.parent
MODEL_DIR = BASE_DIR / "model"

PRODUCTION_MODEL_PATH = MODEL_DIR / "lightgbm_model.txt"
BUNDLED_MODEL_PATH = MODEL_DIR / "bundled_lightgbm_model.txt"
MODEL_METADATA_PATH = MODEL_DIR / "model_metadata.json"


DEFAULT_PRODUCTION_METADATA = {
    "display_name": "Dynamic AMI 24-hour Mortality Model",
    "short_name": "dynamic_ami_24h",
    "version": "1.0.0",
    "alert_threshold": 0.10,
    "description": (
        "Production landmark-level dynamic mortality risk model for critically ill AMI patients."
    ),
    "intended_use": "Research use with validated landmark-level features.",
}

DEFAULT_BUNDLED_METADATA = {
    "display_name": "Bundled Dynamic AMI Model",
    "short_name": "bundled_dynamic_ami_24h",
    "version": "bundle-0.1.0",
    "alert_threshold": 0.10,
    "description": (
        "Bundled LightGBM model file included with the public repository for package checks."
    ),
    "intended_use": (
        "Open-source package validation only. This is not the paper model."
    ),
}

DEFAULT_REFERENCE_METADATA = {
    "display_name": "Reference Fallback Scorer",
    "short_name": "reference_fallback",
    "version": "fallback-0.1.0",
    "alert_threshold": 0.10,
    "description": (
        "Hard-coded fallback scorer used only if neither a production nor bundled model file is available."
    ),
    "intended_use": "Emergency fallback only.",
}


def get_production_model_path() -> Path:
    configured = os.getenv("AMI_MODEL_PATH")
    if configured:
        return Path(configured)
    return PRODUCTION_MODEL_PATH


def get_bundled_model_path() -> Path:
    configured = os.getenv("AMI_BUNDLED_MODEL_PATH")
    if configured:
        return Path(configured)
    return BUNDLED_MODEL_PATH


def _default_metadata(profile: str) -> dict:
    if profile == "production":
        return dict(DEFAULT_PRODUCTION_METADATA)
    if profile == "bundled":
        return dict(DEFAULT_BUNDLED_METADATA)
    return dict(DEFAULT_REFERENCE_METADATA)


def load_model_metadata(profile: str) -> dict:
    metadata = _default_metadata(profile)
    if MODEL_METADATA_PATH.exists():
        try:
            metadata.update(json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            pass

    if "ALERT_THRESHOLD" in os.environ:
        try:
            metadata["alert_threshold"] = float(os.environ["ALERT_THRESHOLD"])
        except ValueError:
            pass

    return metadata
