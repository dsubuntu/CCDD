"""Counterfactual Context Divergence Decoding."""

from .core import (
    DecodingResult,
    TokenDecision,
    decode,
    divergence_scores,
    log_softmax,
    mean_selected_divergence,
    select_token,
)
from .metrics import FaithfulnessMetrics, faithfulness_metrics, normalize_answer

__all__ = [
    "DecodingResult",
    "FaithfulnessMetrics",
    "TokenDecision",
    "decode",
    "divergence_scores",
    "faithfulness_metrics",
    "log_softmax",
    "mean_selected_divergence",
    "normalize_answer",
    "select_token",
]

__version__ = "0.1.0"

