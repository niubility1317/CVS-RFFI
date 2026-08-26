"""Stateless, query-read-only prediction for an SF-TAPFT clean-single model."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class SFTAPFTPrediction:
    logits: Tensor
    predictions: Tensor
    query_truth_opened: bool
    query_role_opened: bool


def _forward_aux(model: nn.Module, values: Tensor) -> Mapping[str, Any]:
    parameters = inspect.signature(model.forward).parameters
    kwargs: dict[str, Any] = {}
    if "return_aux" in parameters:
        kwargs["return_aux"] = True
    for label_name in ("y", "y_tx"):
        if label_name in parameters:
            kwargs[label_name] = None
            break
    outputs = model(values, **kwargs)
    if not isinstance(outputs, Mapping):
        raise ValueError("SF-TAPFT model must return an auxiliary mapping")
    return outputs


def _extract_embedding(outputs: Mapping[str, Any], batch_size: int) -> Tensor:
    nested = outputs.get("aux_id")
    if isinstance(nested, Mapping):
        value = nested.get("feat_joint")
        if torch.is_tensor(value) and value.ndim == 2 and value.size(0) == batch_size:
            return value
    for name in ("feat_joint", "z_id", "feat_cls", "embedding"):
        value = outputs.get(name)
        if torch.is_tensor(value) and value.ndim == 2 and value.size(0) == batch_size:
            return value
    raise ValueError("model output must expose a row-aligned identity embedding")


def _model_device_dtype(model: nn.Module, received_iq: Tensor) -> tuple[torch.device, torch.dtype]:
    for value in tuple(model.parameters()) + tuple(model.buffers()):
        if value.is_floating_point():
            return value.device, value.dtype
    return received_iq.device, received_iq.dtype


def predict_sf_tapft_rows(model, head, received_iq):
    """Predict each received-IQ row independently over every registered class."""

    if not torch.is_tensor(received_iq) or received_iq.ndim < 2 or received_iq.size(0) <= 0:
        raise ValueError("received_iq must be a non-empty batched tensor")
    class_ids = tuple(int(value) for value in head.class_ids)
    if not class_ids:
        raise ValueError("head must expose at least one registered class")

    model.eval()
    head.eval()
    device, dtype = _model_device_dtype(model, received_iq)
    row_logits: list[Tensor] = []
    with torch.no_grad():
        for index in range(int(received_iq.size(0))):
            row = received_iq[index : index + 1].to(device=device, dtype=dtype)
            embedding = _extract_embedding(_forward_aux(model, row), batch_size=1)
            logits = head(embedding)
            if logits.ndim != 2 or logits.shape != (1, len(class_ids)):
                raise ValueError("head logits must cover every registered class for each row")
            row_logits.append(logits)
        logits = torch.cat(row_logits, dim=0).detach().cpu()
        class_id_tensor = torch.tensor(class_ids, dtype=torch.long)
        predictions = class_id_tensor[logits.argmax(dim=1)].detach().cpu()
    return SFTAPFTPrediction(
        logits=logits,
        predictions=predictions,
        query_truth_opened=False,
        query_role_opened=False,
    )
