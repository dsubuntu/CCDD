"""Offline reviewer check: no GPU, network, model, or third-party package."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ccdd.core import decode, mean_selected_divergence  # noqa: E402
from ccdd.datasets import load_benchmark  # noqa: E402
from ccdd.metrics import faithfulness_metrics  # noqa: E402


def _logits(probabilities):
    return tuple(math.log(float(value)) for value in probabilities)


def main() -> int:
    cases = json.loads((ROOT / "data" / "reviewer_cases.json").read_text(encoding="utf-8"))
    checked = []
    for case in cases:
        steps = [
            (_logits(step["factual_probabilities"]), _logits(step["counterfactual_probabilities"]))
            for step in case["steps"]
        ]

        def provider(generated, case_steps=steps):
            return case_steps[min(len(generated), len(case_steps) - 1)]

        result = decode(
            provider,
            max_new_tokens=len(steps) + 2,
            eos_token_id=case["eos_token_id"],
            objective="absolute",
        )
        expected = tuple(case["expected_token_ids"])
        if result.token_ids != expected:
            raise AssertionError(
                f"{case['id']}: expected token ids {expected}, got {result.token_ids}"
            )
        checked.append(
            {
                "id": case["id"],
                "token_ids": result.token_ids,
                "mean_selected_divergence": mean_selected_divergence(result.decisions),
            }
        )

    confiqa = load_benchmark(
        str(ROOT / "data" / "examples" / "confiqa_toy.json"), "confiqa"
    )
    bbq = load_benchmark(str(ROOT / "data" / "examples" / "bbq_age_toy.json"), "bbq_age")
    if len(confiqa) != 2 or len(bbq) != 2:
        raise AssertionError("dataset adapters did not load the expected examples")

    metrics = faithfulness_metrics(
        ["Lorne Balfe", "Mercury"],
        [("Lorne Balfe", "Lorne David Roderick Balfe"), "Venus"],
        ["Petri Alanko", "Venus"],
        ["Petri Alanko", "Venus"],
        parametric_predictions=["Lorne Balfe", "Mercury"],
    )
    expected_metrics = {"CA": 0.5, "CCA": 1.0, "NCA": 0.75, "PA": 1.0}
    actual = metrics.as_dict()
    for key, expected in expected_metrics.items():
        if actual[key] != expected:
            raise AssertionError(f"{key}: expected {expected}, got {actual[key]}")

    print(
        json.dumps(
            {
                "status": "PASS",
                "equation_6_cases": checked,
                "dataset_adapters": {"confiqa": len(confiqa), "bbq_age": len(bbq)},
                "metrics": actual,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

