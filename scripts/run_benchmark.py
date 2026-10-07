"""Small, transparent ConFiQA/BBQ benchmark runner for local HF models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ccdd.core import mean_selected_divergence  # noqa: E402
from ccdd.datasets import load_benchmark  # noqa: E402
from ccdd.hf import generate_ccdd, generate_greedy  # noqa: E402
from ccdd.metrics import faithfulness_metrics  # noqa: E402
from ccdd.prompts import format_qa_prompt  # noqa: E402


def load_model(model_name: str, cpu: bool, trust_remote_code: bool):
    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Install optional dependencies first: python -m pip install -e '.[hf]'"
        ) from exc

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model_class = AutoModelForSeq2SeqLM if config.is_encoder_decoder else AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code
    )
    model = model_class.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model.to(torch.device("cpu" if cpu or not torch.cuda.is_available() else "cuda"))
    return model, tokenizer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="local model path or Hub id")
    parser.add_argument("--data", required=True)
    parser.add_argument("--benchmark", required=True, choices=("confiqa", "bbq_age"))
    parser.add_argument("--method", choices=("standard", "ccdd"), default="ccdd")
    parser.add_argument("--objective", choices=("absolute", "signed"), default="absolute")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    examples = load_benchmark(args.data, args.benchmark, args.limit)
    model, tokenizer = load_model(args.model, args.cpu, args.trust_remote_code)
    records = []
    for example in examples:
        factual_prompt = format_qa_prompt(
            example.factual_context, example.question, choices=example.choices
        )
        counterfactual_prompt = format_qa_prompt(
            example.counterfactual_context, example.question, choices=example.choices
        )
        no_context_prompt = format_qa_prompt(None, example.question, choices=example.choices)
        parametric_prediction = generate_greedy(
            model, tokenizer, no_context_prompt, max_new_tokens=args.max_new_tokens
        )

        factual_divergence = None
        counterfactual_divergence = None
        if args.method == "standard":
            factual_prediction = generate_greedy(
                model, tokenizer, factual_prompt, max_new_tokens=args.max_new_tokens
            )
            counterfactual_prediction = generate_greedy(
                model, tokenizer, counterfactual_prompt, max_new_tokens=args.max_new_tokens
            )
        else:
            factual_result = generate_ccdd(
                model,
                tokenizer,
                factual_prompt,
                counterfactual_prompt,
                max_new_tokens=args.max_new_tokens,
                objective=args.objective,
            )
            counterfactual_result = generate_ccdd(
                model,
                tokenizer,
                counterfactual_prompt,
                factual_prompt,
                max_new_tokens=args.max_new_tokens,
                objective=args.objective,
            )
            factual_prediction = factual_result.text
            counterfactual_prediction = counterfactual_result.text
            factual_divergence = mean_selected_divergence(factual_result.decisions)
            counterfactual_divergence = mean_selected_divergence(
                counterfactual_result.decisions
            )

        record = {
            "id": example.example_id,
            "factual_prediction": factual_prediction,
            "factual_target": example.factual_answers,
            "counterfactual_prediction": counterfactual_prediction,
            "counterfactual_target": example.counterfactual_answers,
            "parametric_prediction": parametric_prediction,
            "factual_mean_selected_divergence": factual_divergence,
            "counterfactual_mean_selected_divergence": counterfactual_divergence,
        }
        records.append(record)
        print(json.dumps(record, ensure_ascii=False))

    metrics = faithfulness_metrics(
        [record["factual_prediction"] for record in records],
        [record["factual_target"] for record in records],
        [record["counterfactual_prediction"] for record in records],
        [record["counterfactual_target"] for record in records],
        parametric_predictions=[record["parametric_prediction"] for record in records],
    )
    output = {
        "configuration": {
            "model": args.model,
            "benchmark": args.benchmark,
            "method": args.method,
            "objective": args.objective if args.method == "ccdd" else None,
            "limit": args.limit,
            "max_new_tokens": args.max_new_tokens,
        },
        "summary": metrics.as_dict(),
        "records": records,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(output["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

