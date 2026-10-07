"""Compute paper metrics from saved predictions without loading a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ccdd.metrics import faithfulness_metrics  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", help="JSONL file following docs/DATA_FORMAT.md")
    args = parser.parse_args()
    records = [
        json.loads(line)
        for line in Path(args.predictions).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    required = {
        "factual_prediction",
        "factual_target",
        "counterfactual_prediction",
        "counterfactual_target",
    }
    for index, record in enumerate(records):
        missing = sorted(required - record.keys())
        if missing:
            raise ValueError(f"record {index} is missing fields: {missing}")

    parametric = (
        [record["parametric_prediction"] for record in records]
        if records and all("parametric_prediction" in record for record in records)
        else None
    )
    metrics = faithfulness_metrics(
        [record["factual_prediction"] for record in records],
        [record["factual_target"] for record in records],
        [record["counterfactual_prediction"] for record in records],
        [record["counterfactual_target"] for record in records],
        parametric_predictions=parametric,
    )
    print(json.dumps(metrics.as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

