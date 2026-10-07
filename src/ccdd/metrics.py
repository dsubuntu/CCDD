"""Context-faithfulness metrics from the accompanying paper."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
import unicodedata
from typing import Callable, Dict, Iterable, Optional, Sequence, Union


Answer = Union[str, int, float]
Target = Union[Answer, Sequence[Answer]]
Matcher = Callable[[Answer, Target], bool]


@dataclass(frozen=True)
class FaithfulnessMetrics:
    """CA, CCA, NCA and optional PA with the evaluated sample count."""

    ca: float
    cca: float
    nca: float
    pa: Optional[float]
    count: int

    def as_dict(self) -> Dict[str, Optional[float]]:
        values = asdict(self)
        values["CA"] = values.pop("ca")
        values["CCA"] = values.pop("cca")
        values["NCA"] = values.pop("nca")
        values["PA"] = values.pop("pa")
        return values


def normalize_answer(value: Answer) -> str:
    """Normalize an answer for reproducible exact-match evaluation."""

    text = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def answer_matches(prediction: Answer, target: Target) -> bool:
    """Match against one answer or a collection of accepted aliases."""

    if isinstance(target, (list, tuple, set, frozenset)):
        accepted: Iterable[Answer] = target
    else:
        accepted = (target,)
    normalized_prediction = normalize_answer(prediction)
    return any(normalized_prediction == normalize_answer(item) for item in accepted)


def _validate_lengths(name: str, values: Sequence[object], expected: int) -> None:
    if len(values) != expected:
        raise ValueError(f"{name} has length {len(values)}; expected {expected}")


def _accuracy(
    predictions: Sequence[Answer],
    targets: Sequence[Target],
    matcher: Matcher,
) -> float:
    if not predictions:
        return 0.0
    return sum(matcher(prediction, target) for prediction, target in zip(predictions, targets)) / len(
        predictions
    )


def faithfulness_metrics(
    factual_predictions: Sequence[Answer],
    factual_targets: Sequence[Target],
    counterfactual_predictions: Sequence[Answer],
    counterfactual_targets: Sequence[Target],
    *,
    parametric_predictions: Optional[Sequence[Answer]] = None,
    matcher: Matcher = answer_matches,
) -> FaithfulnessMetrics:
    """Calculate CA, CCA, NCA=(CA+CCA)/2, and optional PA.

    PA compares factual-condition outputs with answers obtained without context,
    matching the paper's definition P(Y_hat = Y_ParaK).
    """

    count = len(factual_predictions)
    _validate_lengths("factual_targets", factual_targets, count)
    _validate_lengths("counterfactual_predictions", counterfactual_predictions, count)
    _validate_lengths("counterfactual_targets", counterfactual_targets, count)
    if parametric_predictions is not None:
        _validate_lengths("parametric_predictions", parametric_predictions, count)

    ca = _accuracy(factual_predictions, factual_targets, matcher)
    cca = _accuracy(counterfactual_predictions, counterfactual_targets, matcher)
    pa = (
        _accuracy(factual_predictions, parametric_predictions, matcher)
        if parametric_predictions is not None
        else None
    )
    return FaithfulnessMetrics(ca=ca, cca=cca, nca=(ca + cca) / 2.0, pa=pa, count=count)

