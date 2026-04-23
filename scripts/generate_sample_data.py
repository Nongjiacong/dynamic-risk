from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

SEED_DATA_PATH = BASE_DIR / "data" / "sample_seed_profiles.csv"
SAMPLE_OUTPUT_PATH = BASE_DIR / "data" / "sample_landmark_features.csv"
TRAIN_OUTPUT_PATH = BASE_DIR / "data" / "sample_training_landmarks.csv"
RANDOM_SEED = 20260418

NUMERIC_LIMITS = {
    "age": (18, 100),
    "heart_rate_latest": (45, 170),
    "map_latest": (35, 120),
    "respiratory_rate_latest": (8, 45),
    "spo2_latest": (75, 100),
    "gcs_total_latest": (3, 15),
    "urine_output_24h": (0, 3500),
    "lactate_latest": (0.2, 12.0),
    "creatinine_latest": (0.2, 8.0),
    "bun_latest": (4.0, 90.0),
    "wbc_latest": (1.0, 35.0),
    "hemoglobin_latest": (6.0, 18.0),
    "platelet_latest": (20.0, 500.0),
    "glucose_latest": (40.0, 500.0),
    "measurement_count_24h": (1, 24),
    "hours_since_last_measurement": (0.1, 12.0),
}

NOISE_SCALES = {
    "age": 2.0,
    "heart_rate_latest": 6.0,
    "map_latest": 5.0,
    "respiratory_rate_latest": 2.5,
    "spo2_latest": 1.2,
    "gcs_total_latest": 0.7,
    "urine_output_24h": 160.0,
    "lactate_latest": 0.45,
    "creatinine_latest": 0.20,
    "bun_latest": 3.6,
    "wbc_latest": 1.5,
    "hemoglobin_latest": 0.55,
    "platelet_latest": 18.0,
    "glucose_latest": 15.0,
    "measurement_count_24h": 1.6,
    "hours_since_last_measurement": 0.6,
}

BINARY_COLUMNS = [
    "mechanical_ventilation_current",
    "vasoactive_support_current",
    "renal_replacement_therapy_current",
]

TREND_COLUMNS_POSITIVE = {
    "heart_rate_latest": 12.0,
    "map_latest": -11.0,
    "respiratory_rate_latest": 6.0,
    "spo2_latest": -3.0,
    "gcs_total_latest": -2.2,
    "urine_output_24h": -420.0,
    "lactate_latest": 2.0,
    "creatinine_latest": 0.7,
    "bun_latest": 7.0,
    "wbc_latest": 2.0,
    "hemoglobin_latest": -0.7,
    "platelet_latest": -16.0,
    "glucose_latest": 18.0,
    "measurement_count_24h": 3.0,
    "hours_since_last_measurement": -0.7,
}

TREND_COLUMNS_NEGATIVE = {
    "heart_rate_latest": -4.0,
    "map_latest": 4.0,
    "respiratory_rate_latest": -2.0,
    "spo2_latest": 1.0,
    "gcs_total_latest": 0.4,
    "urine_output_24h": 180.0,
    "lactate_latest": -0.5,
    "creatinine_latest": -0.10,
    "bun_latest": -1.8,
    "wbc_latest": -0.8,
    "hemoglobin_latest": 0.2,
    "platelet_latest": 6.0,
    "glucose_latest": -8.0,
    "measurement_count_24h": -1.0,
    "hours_since_last_measurement": 0.8,
}


def _clip_numeric(column: str, value: float) -> float:
    low, high = NUMERIC_LIMITS[column]
    return float(np.clip(value, low, high))


def _binary_with_progress(base: int, progress: float, label: int, rng: np.random.Generator) -> int:
    probability = float(base)
    if label == 1:
        probability = min(1.0, probability + 0.45 * progress)
    else:
        probability = max(0.0, probability - 0.15 * progress)
    return int(rng.random() < probability)


def _make_landmark_row(
    template: pd.Series,
    patient_id: str,
    landmark_hours: int,
    progress: float,
    label: int,
    positive_window: int,
    step_index: int,
    total_steps: int,
    rng: np.random.Generator,
) -> dict:
    row = {
        "patient_id": patient_id,
        "landmark_hours": landmark_hours,
        "sample_case": template["sample_case"],
        "age": int(round(_clip_numeric("age", float(template["age"]) + rng.normal(0.0, NOISE_SCALES["age"])))),
        "sex": template["sex"],
    }

    trend_map = TREND_COLUMNS_POSITIVE if label == 1 else TREND_COLUMNS_NEGATIVE

    for column in NUMERIC_LIMITS:
        base_value = float(template[column])
        trend = trend_map.get(column, 0.0) * progress
        noise = rng.normal(0.0, NOISE_SCALES[column])
        value = _clip_numeric(column, base_value + trend + noise)
        if column in {"gcs_total_latest", "measurement_count_24h"}:
            row[column] = int(round(value))
        else:
            row[column] = round(value, 4)

    for column in BINARY_COLUMNS:
        row[column] = _binary_with_progress(int(template[column]), progress, label, rng)

    positive_cutoff = max(0, total_steps - positive_window)
    row["observed_death_within_24h"] = int(label == 1 and step_index >= positive_cutoff)
    return row


def _generate_patient_rows(
    template: pd.Series,
    patient_index: int,
    prefix: str,
    max_steps: int,
    rng: np.random.Generator,
) -> list[dict]:
    label = int(template["observed_death_within_24h"])
    step_count = int(rng.integers(4, max_steps + 1))
    start_hour = int(rng.choice([6, 12, 18]))
    patient_id = f"{prefix}{patient_index:04d}"
    positive_window = int(rng.integers(1, min(3, step_count) + 1)) if label == 1 else 0

    rows = []
    for step_index in range(step_count):
        landmark_hours = start_hour + 6 * step_index
        progress = 0.0 if step_count == 1 else step_index / (step_count - 1)
        rows.append(
            _make_landmark_row(
                template=template,
                patient_id=patient_id,
                landmark_hours=landmark_hours,
                progress=progress,
                label=label,
                positive_window=positive_window,
                step_index=step_index,
                total_steps=step_count,
                rng=rng,
            )
        )
    return rows


def generate_sample_datasets(
    sample_patients: int = 32,
    training_patients: int = 700,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_SEED)
    seed_frame = pd.read_csv(SEED_DATA_PATH)
    positive_templates = seed_frame[seed_frame["observed_death_within_24h"] == 1].reset_index(drop=True)
    negative_templates = seed_frame[seed_frame["observed_death_within_24h"] == 0].reset_index(drop=True)

    def build_dataset(patient_count: int, prefix: str, max_steps: int) -> pd.DataFrame:
        rows: list[dict] = []
        positive_count = patient_count // 3
        negative_count = patient_count - positive_count

        for idx in range(negative_count):
            template = negative_templates.iloc[idx % len(negative_templates)]
            rows.extend(_generate_patient_rows(template, idx + 1, prefix, max_steps, rng))

        for idx in range(positive_count):
            template = positive_templates.iloc[idx % len(positive_templates)]
            rows.extend(_generate_patient_rows(template, negative_count + idx + 1, prefix, max_steps, rng))

        frame = pd.DataFrame(rows)
        frame = frame.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
        return frame

    sample_frame = build_dataset(sample_patients, "S", max_steps=5)
    training_frame = build_dataset(training_patients, "T", max_steps=7)

    SAMPLE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample_frame.to_csv(SAMPLE_OUTPUT_PATH, index=False)
    training_frame.to_csv(TRAIN_OUTPUT_PATH, index=False)
    return sample_frame, training_frame


if __name__ == "__main__":
    sample_frame, training_frame = generate_sample_datasets()
    print(f"Saved sample dataset: {SAMPLE_OUTPUT_PATH} ({len(sample_frame)} rows)")
    print(f"Saved training dataset: {TRAIN_OUTPUT_PATH} ({len(training_frame)} rows)")
