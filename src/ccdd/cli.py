"""Command-line interface for the lightweight demo and HF generation."""

from __future__ import annotations

import argparse
import json
import math
from typing import Optional, Sequence

from .core import decode, mean_selected_divergence
from .prompts import format_qa_prompt


def _demo(_: argparse.Namespace) -> int:
    vocabulary = ("Paris", "Lyon", "London", "<eos>")
    steps = (
        (
            tuple(math.log(value) for value in (0.70, 0.10, 0.10, 0.10)),
            tuple(math.log(value) for value in (0.10, 0.30, 0.30, 0.30)),
        ),
        (
            tuple(math.log(value) for value in (0.10, 0.10, 0.10, 0.70)),
            tuple(math.log(value) for value in (0.10, 0.10, 0.10, 0.70)),
        ),
    )

    def provider(generated: tuple) -> tuple:
        return steps[min(len(generated), len(steps) - 1)]

    result = decode(provider, max_new_tokens=4, eos_token_id=3)
    payload = {
        "tokens": [vocabulary[token_id] for token_id in result.token_ids],
        "stopped_on_eos": result.stopped_on_eos,
        "mean_selected_divergence": mean_selected_divergence(result.decisions),
    }
    print(json.dumps(payload, indent=2))
    return 0


def _generate(args: argparse.Namespace) -> int:
    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Install optional dependencies first: python -m pip install -e '.[hf]'"
        ) from exc

    from .hf import generate_ccdd

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model_class = AutoModelForSeq2SeqLM if config.is_encoder_decoder else AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    model = model_class.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model.to(device)

    factual_prompt = format_qa_prompt(args.context, args.question)
    counterfactual_prompt = format_qa_prompt(args.counterfactual_context, args.question)
    result = generate_ccdd(
        model,
        tokenizer,
        factual_prompt,
        counterfactual_prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        objective=args.objective,
    )
    print(result.text)
    if args.trace:
        print(
            json.dumps(
                {
                    "token_ids": result.token_ids,
                    "selected_divergence": [
                        decision.divergence for decision in result.decisions
                    ],
                    "stopped_on_eos": result.stopped_on_eos,
                },
                indent=2,
            )
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ccdd")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="run the dependency-free toy example")
    demo.set_defaults(handler=_demo)

    generate = subparsers.add_parser("generate", help="run CCDD with a Hugging Face model")
    generate.add_argument("--model", required=True, help="local model path or Hub id")
    generate.add_argument("--context", required=True)
    generate.add_argument("--counterfactual-context", required=True)
    generate.add_argument("--question", required=True)
    generate.add_argument("--max-new-tokens", type=int, default=32)
    generate.add_argument("--temperature", type=float, default=1.0)
    generate.add_argument("--objective", choices=("absolute", "signed"), default="absolute")
    generate.add_argument("--cpu", action="store_true")
    generate.add_argument("--trust-remote-code", action="store_true")
    generate.add_argument("--trace", action="store_true")
    generate.set_defaults(handler=_generate)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))

