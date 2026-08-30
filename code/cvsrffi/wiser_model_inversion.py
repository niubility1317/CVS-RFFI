"""Isolated non-formal source-head IQ inversion for WISER-RF route C."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelInversionResult:
    pseudo_iq: torch.Tensor
    class_ids: torch.Tensor
    audit: Mapping[str, Any]


def _extract_logits(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, Mapping):
        for key in ("tx_logits", "logits"):
            value = output.get(key)
            if torch.is_tensor(value):
                return value
    raise ValueError("frozen source model did not return classification logits")


def invert_source_head_iq(
    model: nn.Module,
    *,
    class_ids: Sequence[int],
    samples_per_class: int,
    input_channels: int,
    input_length: int,
    steps: int,
    learning_rate: float,
    seed: int,
    target_rms: float = 0.25,
) -> ModelInversionResult:
    """Optimize random bounded IQ against a frozen source classifier.

    This diagnostic API intentionally has no source sample, loader or feature
    argument.  Its output is not eligible for the formal Phase2 bundle.
    """

    classes = tuple(int(value) for value in class_ids)
    if not classes or len(set(classes)) != len(classes) or min(classes) < 0:
        raise ValueError("class_ids must be unique nonnegative integers")
    if int(samples_per_class) < 1:
        raise ValueError("samples_per_class must be positive")
    if int(input_channels) < 1 or int(input_length) < 1:
        raise ValueError("input shape must be positive")
    if int(steps) < 1 or float(learning_rate) <= 0.0:
        raise ValueError("steps and learning_rate must be positive")
    if not 0.0 < float(target_rms) <= 1.0:
        raise ValueError("target_rms must be in (0,1]")

    parameters = list(model.parameters())
    device = parameters[0].device if parameters else torch.device("cpu")
    original_training = bool(model.training)
    original_requires_grad = [parameter.requires_grad for parameter in parameters]
    for parameter in parameters:
        parameter.requires_grad_(False)
    model.eval()

    labels = torch.tensor(
        [value for value in classes for _ in range(int(samples_per_class))],
        device=device,
        dtype=torch.long,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    initial = torch.randn(
        (labels.numel(), int(input_channels), int(input_length)),
        generator=generator,
        dtype=torch.float32,
    ).to(device)
    raw = nn.Parameter(initial * 0.10)
    optimizer = torch.optim.Adam((raw,), lr=float(learning_rate))
    try:
        for _ in range(int(steps)):
            optimizer.zero_grad(set_to_none=True)
            pseudo = torch.tanh(raw)
            logits = _extract_logits(model(pseudo))
            if logits.ndim != 2 or logits.shape[0] != labels.numel():
                raise ValueError("source classifier logits do not align with inversion batch")
            if int(labels.max()) >= int(logits.shape[1]):
                raise ValueError("requested inversion class is absent from source head")
            rms = pseudo.square().mean(dim=(1, 2)).add(1e-8).sqrt()
            smoothness = (pseudo[..., 1:] - pseudo[..., :-1]).square().mean()
            loss = (
                F.cross_entropy(logits, labels)
                + 2.0 * (rms - float(target_rms)).square().mean()
                + 0.01 * smoothness
            )
            loss.backward()
            optimizer.step()
        pseudo_iq = torch.tanh(raw).detach().cpu()
        labels_cpu = labels.detach().cpu()
    finally:
        for parameter, requires_grad in zip(parameters, original_requires_grad):
            parameter.requires_grad_(requires_grad)
        model.train(original_training)

    return ModelInversionResult(
        pseudo_iq=pseudo_iq,
        class_ids=labels_cpu,
        audit={
            "schema": "cvs.wiser.model_inversion_diagnostic.v1",
            "status": "DIAGNOSTIC_MODEL_INVERSION_NON_FORMAL",
            "formal_phase2_eligible": False,
            "source_sample_access": False,
            "source_feature_access": False,
            "initialization": "deterministic_random_noise",
            "seed": int(seed),
            "steps": int(steps),
            "samples_per_class": int(samples_per_class),
        },
    )


__all__ = ["ModelInversionResult", "invert_source_head_iq"]
