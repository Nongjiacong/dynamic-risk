from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dynamic_ami_risk import (
    DEFAULT_LABEL_COLUMN,
    build_production_metadata,
    fit_lightgbm_model,
    prepare_training_frame,
    save_model_file,
    write_metadata,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a LightGBM model file from a private landmark-level CSV."
    )
    parser.add_argument("--input", required=True, help="Path to the training CSV.")
    parser.add_argument(
        "--output-model",
        default="model/lightgbm_model.txt",
        help="Where to save the trained LightGBM model file.",
    )
    parser.add_argument(
        "--label-column",
        default=DEFAULT_LABEL_COLUMN,
        help="Binary outcome column in the training CSV.",
    )
    parser.add_argument(
        "--metadata-json",
        default=None,
        help="Optional path to write production model metadata JSON.",
    )
    parser.add_argument(
        "--display-name",
        default=None,
        help="Optional display name for the metadata JSON.",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Optional version for the metadata JSON.",
    )
    parser.add_argument(
        "--alert-threshold",
        type=float,
        default=None,
        help="Optional alert threshold for the metadata JSON.",
    )
    parser.add_argument(
        "--description",
        default=None,
        help="Optional description for the metadata JSON.",
    )
    parser.add_argument(
        "--intended-use",
        default=None,
        help="Optional intended-use text for the metadata JSON.",
    )
    parser.add_argument(
        "--num-boost-round",
        type=int,
        default=180,
        help="Number of LightGBM boosting rounds.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260418,
        help="Random seed for LightGBM training.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    output_model = Path(args.output_model)
    metadata_path = Path(args.metadata_json) if args.metadata_json else None

    frame = pd.read_csv(input_path)
    features, labels = prepare_training_frame(frame, label_column=args.label_column)
    booster = fit_lightgbm_model(
        features,
        labels,
        num_boost_round=args.num_boost_round,
        random_seed=args.seed,
    )
    save_model_file(booster, output_model)

    print(f"Saved model file to {output_model}")
    print(f"Training rows: {len(frame)}")
    print(f"Positive rate: {float(labels.mean()):.4f}")

    if metadata_path is not None:
        metadata = build_production_metadata(
            display_name=args.display_name,
            version=args.version,
            alert_threshold=args.alert_threshold,
            description=args.description,
            intended_use=args.intended_use,
        )
        write_metadata(metadata_path, metadata)
        print(f"Saved metadata to {metadata_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
