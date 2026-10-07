"""Prompt templates shared by the examples and benchmark runner."""

from __future__ import annotations

from typing import Mapping, Optional


def format_qa_prompt(
    context: Optional[str],
    question: str,
    *,
    choices: Optional[Mapping[str, str]] = None,
) -> str:
    """Create the compact QA prompt used by the cleaned evaluation code."""

    sections = []
    if context:
        sections.append(f"Context: {context.strip()}")
    sections.append(f"Question: {question.strip()}")
    if choices:
        rendered = "\n".join(f"{key}: {value}" for key, value in choices.items())
        sections.append(f"Options:\n{rendered}")
        sections.append("Provide the answer as one option number only:")
    else:
        sections.append("Provide the shortest possible answer:")
    return "\n".join(sections)

