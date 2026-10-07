"""Adapters for the ConFiQA-QA and BBQ-Age schemas used in the paper."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ConflictExample:
    example_id: str
    question: str
    factual_context: str
    counterfactual_context: str
    factual_answers: Tuple[str, ...]
    counterfactual_answers: Tuple[str, ...]
    choices: Optional[Dict[str, str]] = None


def _read_json_or_jsonl(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as stream:
        if path.suffix.lower() == ".jsonl":
            return [json.loads(line) for line in stream if line.strip()]
        data = json.load(stream)
    if not isinstance(data, list):
        raise ValueError("dataset root must be a JSON array or JSONL records")
    return data


def _require(record: dict, fields: Iterable[str], index: int) -> None:
    missing = [field for field in fields if field not in record]
    if missing:
        raise ValueError(f"record {index} is missing required fields: {missing}")


def load_confiqa(path: str, limit: Optional[int] = None) -> List[ConflictExample]:
    records = _read_json_or_jsonl(Path(path))
    examples = []
    for index, record in enumerate(records[:limit] if limit is not None else records):
        _require(
            record,
            ("question", "orig_context", "cf_context", "orig_answer", "cf_answer"),
            index,
        )
        factual = [str(record["orig_answer"])]
        factual.extend(str(item) for item in record.get("orig_alias", []))
        counterfactual = [str(record["cf_answer"])]
        counterfactual.extend(str(item) for item in record.get("cf_alias", []))
        examples.append(
            ConflictExample(
                example_id=str(record.get("id", record.get("orig_path", index))),
                question=str(record["question"]),
                factual_context=str(record["orig_context"]),
                counterfactual_context=str(record["cf_context"]),
                factual_answers=tuple(factual),
                counterfactual_answers=tuple(counterfactual),
            )
        )
    return examples


def load_bbq_age(path: str, limit: Optional[int] = None) -> List[ConflictExample]:
    records = _read_json_or_jsonl(Path(path))
    examples = []
    for index, record in enumerate(records[:limit] if limit is not None else records):
        _require(
            record,
            ("question", "orig_context", "cf_context", "orig_answer", "cf_answer"),
            index,
        )
        choices = {
            str(choice): str(record[f"orig_ans{choice}"])
            for choice in range(3)
            if f"orig_ans{choice}" in record
        }
        if len(choices) != 3:
            raise ValueError(f"record {index} must define orig_ans0, orig_ans1, and orig_ans2")
        factual_index = str(record["orig_answer"])
        counterfactual_index = str(record["cf_answer"])
        examples.append(
            ConflictExample(
                example_id=str(record.get("example_id", index)),
                question=str(record["question"]),
                factual_context=str(record["orig_context"]),
                counterfactual_context=str(record["cf_context"]),
                factual_answers=(factual_index, choices[factual_index]),
                counterfactual_answers=(counterfactual_index, choices[counterfactual_index]),
                choices=choices,
            )
        )
    return examples


def load_benchmark(
    path: str,
    benchmark: str,
    limit: Optional[int] = None,
) -> Sequence[ConflictExample]:
    normalized = benchmark.casefold().replace("-", "_")
    if normalized in {"confiqa", "confiqa_qa"}:
        return load_confiqa(path, limit)
    if normalized in {"bbq", "bbq_age"}:
        return load_bbq_age(path, limit)
    raise ValueError("benchmark must be 'confiqa' or 'bbq_age'")

