"""Optional Hugging Face adapter for decoder-only and encoder-decoder models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .core import TokenDecision


@dataclass(frozen=True)
class HFGenerationResult:
    text: str
    token_ids: Tuple[int, ...]
    decisions: Tuple[TokenDecision, ...]
    stopped_on_eos: bool


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Hugging Face generation requires the optional dependencies. "
            "Install with: python -m pip install -e '.[hf]'"
        ) from exc
    return torch


def _move_to_device(batch: Dict[str, Any], device: Any) -> Dict[str, Any]:
    return {key: value.to(device) for key, value in batch.items()}


def _eos_ids(model: Any, tokenizer: Any, explicit: Optional[int]) -> set:
    if explicit is not None:
        return {int(explicit)}
    value = getattr(model.config, "eos_token_id", None)
    if value is None:
        value = getattr(tokenizer, "eos_token_id", None)
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {int(item) for item in value}
    return {int(value)}


def _decoder_start_id(model: Any, tokenizer: Any) -> int:
    for value in (
        getattr(model.config, "decoder_start_token_id", None),
        getattr(tokenizer, "pad_token_id", None),
        getattr(tokenizer, "bos_token_id", None),
    ):
        if value is not None:
            return int(value)
    raise ValueError("encoder-decoder model requires a decoder start, pad, or BOS token id")


def _decision_from_logits(
    factual_logits: Any,
    counterfactual_logits: Any,
    *,
    temperature: float,
    objective: str,
) -> Tuple[int, TokenDecision]:
    torch = _require_torch()
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero")
    if objective not in {"absolute", "signed"}:
        raise ValueError("objective must be 'absolute' or 'signed'")

    factual_log_probs = torch.log_softmax(factual_logits.float() / temperature, dim=-1)
    counterfactual_log_probs = torch.log_softmax(
        counterfactual_logits.float() / temperature, dim=-1
    )
    signed = factual_log_probs - counterfactual_log_probs
    scores = signed.abs() if objective == "absolute" else signed

    maximum = scores.max()
    tied = torch.nonzero(scores == maximum, as_tuple=False).flatten()
    if tied.numel() > 1:
        tied_factual = factual_log_probs[tied]
        best_factual = tied_factual.max()
        tied = tied[tied_factual == best_factual]
    token_id = int(tied.min().item())
    signed_value = float(signed[token_id].item())
    decision = TokenDecision(
        token_id=token_id,
        factual_log_probability=float(factual_log_probs[token_id].item()),
        counterfactual_log_probability=float(counterfactual_log_probs[token_id].item()),
        signed_divergence=signed_value,
        divergence=abs(signed_value),
        objective=objective,
    )
    return token_id, decision


def generate_ccdd(
    model: Any,
    tokenizer: Any,
    factual_prompt: str,
    counterfactual_prompt: str,
    *,
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    objective: str = "absolute",
    eos_token_id: Optional[int] = None,
) -> HFGenerationResult:
    """Generate with paired contexts and a shared autoregressive token history."""

    torch = _require_torch()
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    device = next(model.parameters()).device
    factual_inputs = _move_to_device(tokenizer(factual_prompt, return_tensors="pt"), device)
    counterfactual_inputs = _move_to_device(
        tokenizer(counterfactual_prompt, return_tensors="pt"), device
    )
    is_encoder_decoder = bool(getattr(model.config, "is_encoder_decoder", False))
    eos_ids = _eos_ids(model, tokenizer, eos_token_id)
    generated: List[int] = []
    decisions: List[TokenDecision] = []
    stopped_on_eos = False

    if is_encoder_decoder:
        decoder_ids = torch.tensor(
            [[_decoder_start_id(model, tokenizer)]], dtype=torch.long, device=device
        )

    model.eval()
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            if is_encoder_decoder:
                factual_output = model(**factual_inputs, decoder_input_ids=decoder_ids)
                counterfactual_output = model(
                    **counterfactual_inputs, decoder_input_ids=decoder_ids
                )
            else:
                if generated:
                    suffix = torch.tensor([generated], dtype=torch.long, device=device)
                    factual_ids = torch.cat((factual_inputs["input_ids"], suffix), dim=1)
                    counterfactual_ids = torch.cat(
                        (counterfactual_inputs["input_ids"], suffix), dim=1
                    )
                else:
                    factual_ids = factual_inputs["input_ids"]
                    counterfactual_ids = counterfactual_inputs["input_ids"]
                factual_output = model(input_ids=factual_ids)
                counterfactual_output = model(input_ids=counterfactual_ids)

            token_id, decision = _decision_from_logits(
                factual_output.logits[0, -1],
                counterfactual_output.logits[0, -1],
                temperature=temperature,
                objective=objective,
            )
            generated.append(token_id)
            decisions.append(decision)
            if is_encoder_decoder:
                next_token = torch.tensor([[token_id]], dtype=torch.long, device=device)
                decoder_ids = torch.cat((decoder_ids, next_token), dim=1)
            if token_id in eos_ids:
                stopped_on_eos = True
                break

    text = tokenizer.decode(generated, skip_special_tokens=True)
    return HFGenerationResult(text, tuple(generated), tuple(decisions), stopped_on_eos)


def generate_greedy(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int = 32,
) -> str:
    """Deterministic baseline generation used by the benchmark helper."""

    torch = _require_torch()
    device = next(model.parameters()).device
    inputs = _move_to_device(tokenizer(prompt, return_tensors="pt"), device)
    model.eval()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=getattr(tokenizer, "pad_token_id", None),
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
        )
    if bool(getattr(model.config, "is_encoder_decoder", False)):
        generated = output[0]
    else:
        generated = output[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()

