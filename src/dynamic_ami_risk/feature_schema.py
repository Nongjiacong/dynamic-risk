from dataclasses import asdict, dataclass
from typing import List


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    dtype: str
    description: str


REQUIRED_FEATURES: List[FeatureDefinition] = [
    FeatureDefinition("age", "numeric", "Age in years at or before the landmark."),
    FeatureDefinition("sex", "string_or_binary", "Patient sex. Accepted values include male/female or 1/0."),
    FeatureDefinition("heart_rate_latest", "numeric", "Most recent heart rate in beats per minute."),
    FeatureDefinition("map_latest", "numeric", "Most recent mean arterial pressure in mmHg."),
    FeatureDefinition("respiratory_rate_latest", "numeric", "Most recent respiratory rate in breaths per minute."),
    FeatureDefinition("spo2_latest", "numeric", "Most recent peripheral oxygen saturation in percent."),
    FeatureDefinition("gcs_total_latest", "numeric", "Most recent Glasgow Coma Scale total score."),
    FeatureDefinition("urine_output_24h", "numeric", "Total urine output over the prior 24 hours in mL."),
    FeatureDefinition("lactate_latest", "numeric", "Most recent lactate in mmol/L."),
    FeatureDefinition("creatinine_latest", "numeric", "Most recent creatinine in mg/dL."),
    FeatureDefinition("bun_latest", "numeric", "Most recent blood urea nitrogen in mg/dL."),
    FeatureDefinition("wbc_latest", "numeric", "Most recent white blood cell count in 10^9/L."),
    FeatureDefinition("hemoglobin_latest", "numeric", "Most recent hemoglobin in g/dL."),
    FeatureDefinition("platelet_latest", "numeric", "Most recent platelet count in 10^9/L."),
    FeatureDefinition("glucose_latest", "numeric", "Most recent glucose in mg/dL."),
    FeatureDefinition("mechanical_ventilation_current", "binary", "Current invasive mechanical ventilation at the landmark."),
    FeatureDefinition("vasoactive_support_current", "binary", "Current vasoactive support at the landmark."),
    FeatureDefinition("renal_replacement_therapy_current", "binary", "Current renal replacement therapy at the landmark."),
    FeatureDefinition("measurement_count_24h", "numeric", "Number of measurements captured during the prior 24 hours."),
    FeatureDefinition("hours_since_last_measurement", "numeric", "Elapsed hours since the most recent measurement."),
]

OPTIONAL_COLUMNS = [
    "patient_id",
    "landmark_hours",
    "sample_case",
    "observed_death_within_24h",
]


def required_column_names() -> List[str]:
    return [feature.name for feature in REQUIRED_FEATURES]


def schema_payload() -> dict:
    return {
        "required_features": [asdict(feature) for feature in REQUIRED_FEATURES],
        "optional_columns": OPTIONAL_COLUMNS,
    }
