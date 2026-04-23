from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dynamic_ami_risk import load_model, score_frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score a landmark-level CSV with the dynamic AMI mortality model."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the scored output CSV file.",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional path to write the run summary as JSON.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary_json) if args.summary_json else None

    frame = pd.read_csv(input_path)
    result = score_frame(frame, model=load_model())
    predictions = pd.DataFrame(result["predictions"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)

    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(
                {
                    "model": result["model"],
                    "summary": result["summary"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    print(f"Scored rows: {result['summary']['rows']}")
    print(f"Alerts: {result['summary']['alert_count']}")
    print(f"Mean risk: {result['summary']['mean_risk']:.4f}")
    print(f"Saved scored CSV to {output_path}")
    if summary_path is not None:
        print(f"Saved summary JSON to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
