"""Dependency-free implementation of the CCDD token-selection rule.

The absolute objective implements Eq. (6) in the paper:

    argmax_v |log p(v | c, q, y_<t) - log p(v | c', q, y_<t)|

The optional signed objective implements an oriented contrast:

    argmax_v log p(v | c, q, y_<t) - log p(v | c', q, y_<t)

The latter is useful when c is explicitly the primary context. It is not the
printed Eq. (6), so callers must select it explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable, Optional, Sequence, Tuple


VALID_OBJECTIVES = ("absolute", "signed")


@dataclass(frozen=True)
class TokenDecision:
    """A single, auditable CCDD decoding decision."""

    token_id: int
    factual_log_probability: float
    counterfactual_log_probability: float
    signed_divergence: float
    divergence: float
    objective: str


@dataclass(frozen=True)
class DecodingResult:
    """Generated token ids and the decision trace used to select them."""

    token_ids: Tuple[int, ...]
    decisions: Tuple[TokenDecision, ...]
    stopped_on_eos: bool


StepLogits = Callable[[Tuple[int, ...]], Tuple[Sequence[float], Sequence[float]]]


def _as_finite_floats(values: Sequence[float], name: str) -> Tuple[float, ...]:
    if not values:
        raise ValueError(f"{name} must contain at least one logit")
    converted = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in converted):
        raise ValueError(f"{name} must contain only finite values")
    return converted


def log_softmax(logits: Sequence[float], temperature: float = 1.0) -> Tuple[float, ...]:
    """Compute a numerically stable log-softmax using only the standard library."""

    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be a finite number greater than zero")
    values = _as_finite_floats(logits, "logits")
    scaled = tuple(value / temperature for value in values)
    maximum = max(scaled)
    log_normalizer = maximum + math.log(sum(math.exp(value - maximum) for value in scaled))
    return tuple(value - log_normalizer for value in scaled)


def divergence_scores(
    factual_logits: Sequence[float],
    counterfactual_logits: Sequence[float],
    *,
    temperature: float = 1.0,
    objective: str = "absolute",
) -> Tuple[float, ...]:
    """Return the vocabulary-wide CCDD score for each token."""

    if objective not in VALID_OBJECTIVES:
        raise ValueError(f"objective must be one of {VALID_OBJECTIVES}, got {objective!r}")
    factual = log_softmax(factual_logits, temperature)
    counterfactual = log_softmax(counterfactual_logits, temperature)
    if len(factual) != len(counterfactual):
        raise ValueError("factual and counterfactual logits must have the same length")

    signed = tuple(left - right for left, right in zip(factual, counterfactual))
    if objective == "absolute":
        return tuple(abs(value) for value in signed)
    return signed


def select_token(
    factual_logits: Sequence[float],
    counterfactual_logits: Sequence[float],
    *,
    temperature: float = 1.0,
    objective: str = "absolute",
    candidate_token_ids: Optional[Iterable[int]] = None,
) -> TokenDecision:
    """Select one token using CCDD.

    Ties are resolved by higher factual probability and then by lower token id.
    This makes the result deterministic across Python versions and devices.
    """

    factual = log_softmax(factual_logits, temperature)
    counterfactual = log_softmax(counterfactual_logits, temperature)
    if len(factual) != len(counterfactual):
        raise ValueError("factual and counterfactual logits must have the same length")
    if objective not in VALID_OBJECTIVES:
        raise ValueError(f"objective must be one of {VALID_OBJECTIVES}, got {objective!r}")

    signed = tuple(left - right for left, right in zip(factual, counterfactual))
    scores = tuple(abs(value) for value in signed) if objective == "absolute" else signed

    if candidate_token_ids is None:
        candidates = tuple(range(len(scores)))
    else:
        candidates = tuple(int(token_id) for token_id in candidate_token_ids)
        if not candidates:
            raise ValueError("candidate_token_ids must not be empty")
        invalid = [token_id for token_id in candidates if token_id < 0 or token_id >= len(scores)]
        if invalid:
            raise ValueError(f"candidate token ids out of range: {invalid}")

    selected = max(candidates, key=lambda token_id: (scores[token_id], factual[token_id], -token_id))
    return TokenDecision(
        token_id=selected,
        factual_log_probability=factual[selected],
        counterfactual_log_probability=counterfactual[selected],
        signed_divergence=signed[selected],
        divergence=abs(signed[selected]),
        objective=objective,
    )


def decode(
    step_logits: StepLogits,
    *,
    max_new_tokens: int,
    eos_token_id: Optional[int] = None,
    temperature: float = 1.0,
    objective: str = "absolute",
    candidate_token_ids: Optional[Iterable[int]] = None,
) -> DecodingResult:
    """Run autoregressive CCDD against a model-independent logits callback."""

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    generated = []
    decisions = []
    stopped_on_eos = False
    for _ in range(max_new_tokens):
        factual_logits, counterfactual_logits = step_logits(tuple(generated))
        decision = select_token(
            factual_logits,
            counterfactual_logits,
            temperature=temperature,
            objective=objective,
            candidate_token_ids=candidate_token_ids,
        )
        generated.append(decision.token_id)
        decisions.append(decision)
        if eos_token_id is not None and decision.token_id == eos_token_id:
            stopped_on_eos = True
            break

    return DecodingResult(tuple(generated), tuple(decisions), stopped_on_eos)


def mean_selected_divergence(decisions: Sequence[TokenDecision]) -> float:
    """Compute the dataset/sequence average corresponding to Eqs. (7)-(8)."""

    if not decisions:
        return 0.0
    return sum(decision.divergence for decision in decisions) / len(decisions)

