"""Gradient-isolated episodic rejector for Phase1 source-only development.

The identity representation is treated as immutable.  A low-capacity head is
trained from class-symmetric relative geometry built by leaving one complete
source TX out of every inner episode.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


GI_EPIOR_THRESHOLD = 0.5
GI_EPIOR_SEED = 7281105
GI_EPIOR_EPOCHS = 200
GI_EPIOR_LR = 1.0e-2
GI_EPIOR_WEIGHT_DECAY = 1.0e-3
GI_EPIOR_HIDDEN = 8
GI_EPIOR_EPS = 1.0e-6


class GIEpiORError(ValueError):
    """Raised when the frozen GI-EpiOR contract is violated."""


def _as_strings(values: Sequence[Any]) -> np.ndarray:
    return np.asarray([str(value) for value in values], dtype=object)


def canonical_physical_ids(
    tx_ids: Sequence[Any],
    rx_ids: Sequence[Any],
    day_ids: Sequence[Any],
    eq_ids: Sequence[Any],
    sig_ids: Sequence[Any],
) -> np.ndarray:
    columns = [_as_strings(values) for values in (tx_ids, rx_ids, day_ids, eq_ids, sig_ids)]
    sizes = {int(column.size) for column in columns}
    if len(sizes) != 1:
        raise GIEpiORError("physical metadata columns must have equal length")
    out: list[str] = []
    for row in zip(*(column.tolist() for column in columns)):
        if any(not str(value).strip() for value in row):
            raise GIEpiORError("canonical physical ID rejects empty metadata")
        out.append(json.dumps(list(row), ensure_ascii=False, separators=(",", ":")))
    if len(set(out)) != len(out):
        raise GIEpiORError("canonical physical IDs must be unique")
    return np.asarray(out, dtype=object)


def deterministic_reference_query_split(
    tx_ids: Sequence[Any],
    physical_ids: Sequence[Any],
    source_tx_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    tx = _as_strings(tx_ids)
    physical = _as_strings(physical_ids)
    if tx.size != physical.size:
        raise GIEpiORError("TX and physical ID rows must match")
    source = tuple(str(value) for value in source_tx_ids)
    if len(source) < 3 or len(set(source)) != len(source):
        raise GIEpiORError("GI-EpiOR requires at least three unique source TX")
    reference = np.zeros(tx.size, dtype=bool)
    query = np.zeros(tx.size, dtype=bool)
    per_tx: dict[str, dict[str, int]] = {}
    for class_id in source:
        indices = np.flatnonzero(tx == class_id)
        if indices.size < 4:
            raise GIEpiORError(f"source TX {class_id} requires at least four physical rows")
        ranked = sorted(
            indices.tolist(),
            key=lambda index: (
                hashlib.sha256(str(physical[index]).encode("utf-8")).digest(),
                str(physical[index]),
            ),
        )
        cut = len(ranked) // 2
        ref_rows = np.asarray(ranked[:cut], dtype=np.int64)
        query_rows = np.asarray(ranked[cut:], dtype=np.int64)
        if ref_rows.size == 0 or query_rows.size == 0:
            raise GIEpiORError(f"source TX {class_id} produced an empty split")
        reference[ref_rows] = True
        query[query_rows] = True
        per_tx[class_id] = {"reference": int(ref_rows.size), "query": int(query_rows.size)}
    if bool(np.any(reference & query)):
        raise GIEpiORError("reference/query row overlap")
    ref_ids = set(physical[reference].tolist())
    query_ids = set(physical[query].tolist())
    if ref_ids & query_ids:
        raise GIEpiORError("reference/query physical ID overlap")
    selected = np.isin(tx, np.asarray(source, dtype=object))
    if not np.array_equal(reference | query, selected):
        raise GIEpiORError("source rows are not closed by reference/query split")
    receipt = {
        "schema": "cvs.phase1.gi_epior_split.v1",
        "source_tx_ids": list(source),
        "reference_rows": int(reference.sum()),
        "query_rows": int(query.sum()),
        "physical_overlap": 0,
        "per_tx": per_tx,
    }
    return reference, query, receipt


def _normalized_features(features: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(features, dtype=torch.float32).detach()
    if value.ndim != 2 or value.size(0) < 1 or value.size(1) < 2:
        raise GIEpiORError("features must have shape [N,D]")
    if not bool(torch.isfinite(value).all()):
        raise GIEpiORError("features must be finite")
    norms = torch.linalg.vector_norm(value, dim=1)
    if bool(torch.any(norms <= 0.0)):
        raise GIEpiORError("features must have non-zero norm")
    return F.normalize(value, dim=1)


def fit_class_geometry(
    features: torch.Tensor,
    tx_ids: Sequence[Any],
    reference_mask: Sequence[bool],
    class_ids: Sequence[str],
    *,
    eps: float = GI_EPIOR_EPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    z = _normalized_features(features)
    tx = _as_strings(tx_ids)
    reference = np.asarray(reference_mask, dtype=bool)
    if tx.size != z.size(0) or reference.size != z.size(0):
        raise GIEpiORError("geometry metadata row mismatch")
    prototypes: list[torch.Tensor] = []
    scales: list[torch.Tensor] = []
    for class_id in class_ids:
        rows = np.flatnonzero(reference & (tx == str(class_id)))
        if rows.size < 2:
            raise GIEpiORError(f"class {class_id} requires at least two reference rows")
        class_z = z[torch.as_tensor(rows, dtype=torch.long)]
        prototype = F.normalize(class_z.mean(dim=0, keepdim=True), dim=1)[0]
        distance = 1.0 - class_z @ prototype
        median = torch.median(distance)
        mad = torch.median(torch.abs(distance - median)).clamp_min(float(eps))
        prototypes.append(prototype)
        scales.append(mad)
    if len(prototypes) < 2:
        raise GIEpiORError("GI-EpiOR geometry requires at least two classes")
    return torch.stack(prototypes, dim=0), torch.stack(scales, dim=0)


def geometry_descriptors(
    features: torch.Tensor,
    prototypes: torch.Tensor,
    scales: torch.Tensor,
    *,
    eps: float = GI_EPIOR_EPS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z = _normalized_features(features)
    prototype = F.normalize(torch.as_tensor(prototypes, dtype=torch.float32).detach(), dim=1)
    rho = torch.as_tensor(scales, dtype=torch.float32).detach().reshape(-1)
    if prototype.ndim != 2 or prototype.size(0) < 2 or prototype.size(1) != z.size(1):
        raise GIEpiORError("prototype shape mismatch")
    if rho.numel() != prototype.size(0) or not bool(torch.isfinite(rho).all()) or bool(torch.any(rho <= 0.0)):
        raise GIEpiORError("MAD scales must be finite and positive")
    d_class = (1.0 - z @ prototype.T) / (rho.unsqueeze(0) + float(eps))
    ordered = torch.sort(d_class, dim=1).values
    d1 = ordered[:, 0]
    d2 = ordered[:, 1]
    ratio = d1 / (d2 + float(eps))
    descriptor = torch.stack([d1, d2 - d1, ratio], dim=1)
    if not bool(torch.isfinite(descriptor).all()):
        raise GIEpiORError("geometry descriptors must be finite")
    return descriptor, d_class, ratio


class GIEpiORHead(nn.Module):
    def __init__(self, hidden: int = GI_EPIOR_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, int(hidden)), nn.Tanh(), nn.Linear(int(hidden), 1))

    def forward(self, descriptor: torch.Tensor) -> torch.Tensor:
        return self.net(descriptor).squeeze(-1)


class GIEpiORRuntime(nn.Module):
    def __init__(self, prototypes: torch.Tensor, scales: torch.Tensor, head: GIEpiORHead, eps: float = GI_EPIOR_EPS):
        super().__init__()
        self.register_buffer("prototypes", F.normalize(prototypes.detach().float(), dim=1))
        self.register_buffer("scales", scales.detach().float().reshape(-1))
        self.head = head
        self.eps = float(eps)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = F.normalize(features.float(), dim=1)
        d_class = (1.0 - z @ self.prototypes.T) / (self.scales.unsqueeze(0) + self.eps)
        ordered = torch.sort(d_class, dim=1).values
        d1 = ordered[:, 0]
        d2 = ordered[:, 1]
        ratio = d1 / (d2 + self.eps)
        descriptor = torch.stack([d1, d2 - d1, ratio], dim=1)
        e_epi = torch.sigmoid(self.head(descriptor))
        return e_epi, d_class, ratio


@dataclass
class GIEpiORFitResult:
    runtime: GIEpiORRuntime
    reference_mask: np.ndarray
    query_mask: np.ndarray
    receipt: dict[str, Any]


def fit_gi_epior(
    features: torch.Tensor,
    tx_ids: Sequence[Any],
    physical_ids: Sequence[Any],
    source_tx_ids: Sequence[str],
    *,
    seed: int = GI_EPIOR_SEED,
    epochs: int = GI_EPIOR_EPOCHS,
    lr: float = GI_EPIOR_LR,
    weight_decay: float = GI_EPIOR_WEIGHT_DECAY,
    hidden: int = GI_EPIOR_HIDDEN,
    eps: float = GI_EPIOR_EPS,
) -> GIEpiORFitResult:
    if int(seed) != GI_EPIOR_SEED or int(epochs) != GI_EPIOR_EPOCHS:
        raise GIEpiORError("GI-EpiOR seed and epochs are frozen")
    if float(lr) != GI_EPIOR_LR or float(weight_decay) != GI_EPIOR_WEIGHT_DECAY or int(hidden) != GI_EPIOR_HIDDEN:
        raise GIEpiORError("GI-EpiOR optimizer and head shape are frozen")
    z = _normalized_features(features)
    tx = _as_strings(tx_ids)
    physical = _as_strings(physical_ids)
    source = tuple(str(value) for value in source_tx_ids)
    reference, query, split_receipt = deterministic_reference_query_split(tx, physical, source)
    descriptors: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    episode_rows: dict[str, dict[str, int]] = {}
    for held_tx in source:
        registry = tuple(value for value in source if value != held_tx)
        prototypes, scales = fit_class_geometry(z, tx, reference, registry, eps=eps)
        positive_rows = np.flatnonzero(query & (tx == held_tx))
        negative_rows = np.flatnonzero(query & np.isin(tx, np.asarray(registry, dtype=object)))
        if positive_rows.size == 0 or negative_rows.size == 0:
            raise GIEpiORError(f"episode {held_tx} has empty positive or negative query")
        positive, _, _ = geometry_descriptors(z[torch.as_tensor(positive_rows)], prototypes, scales, eps=eps)
        negative, _, _ = geometry_descriptors(z[torch.as_tensor(negative_rows)], prototypes, scales, eps=eps)
        descriptors.extend([positive, negative])
        labels.extend([torch.ones(positive.size(0)), torch.zeros(negative.size(0))])
        episode_rows[held_tx] = {
            "reference_classes": len(registry),
            "positive_rows": int(positive_rows.size),
            "negative_rows": int(negative_rows.size),
            "held_reference_rows": int(np.sum(reference & (tx == held_tx) & np.isin(tx, np.asarray(registry, dtype=object)))),
        }
    train_x = torch.cat(descriptors, dim=0).detach()
    train_y = torch.cat(labels, dim=0).detach()
    if not bool(torch.isfinite(train_x).all()) or set(train_y.tolist()) != {0.0, 1.0}:
        raise GIEpiORError("episodic training rows are invalid")
    torch.manual_seed(int(seed))
    head = GIEpiORHead(hidden=hidden)
    optimizer = torch.optim.Adam(head.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    positive = float(train_y.sum().item())
    negative = float(train_y.numel() - positive)
    pos_weight = torch.tensor(negative / max(positive, 1.0), dtype=torch.float32)
    final_loss = torch.tensor(float("nan"))
    head_grad_norm = 0.0
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        logits = head(train_x)
        final_loss = F.binary_cross_entropy_with_logits(logits, train_y, pos_weight=pos_weight)
        if not bool(torch.isfinite(final_loss)):
            raise GIEpiORError("episodic BCE became non-finite")
        final_loss.backward()
        head_grad_norm = float(sum(parameter.grad.detach().abs().sum().item() for parameter in head.parameters() if parameter.grad is not None))
        optimizer.step()
    if head_grad_norm <= 0.0:
        raise GIEpiORError("GI-EpiOR head received no gradient")
    prototypes, scales = fit_class_geometry(z, tx, reference, source, eps=eps)
    runtime = GIEpiORRuntime(prototypes, scales, head.eval(), eps=eps).eval()
    receipt = {
        "schema": "cvs.phase1.gi_epior_fit.v1",
        "source_tx_ids": list(source),
        "outer_rows_opened_for_fit": 0,
        "threshold": GI_EPIOR_THRESHOLD,
        "threshold_policy": "fixed_sigmoid_0p5_no_quantile_no_outer_calibration",
        "seed": int(seed),
        "epochs": int(epochs),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "hidden": int(hidden),
        "train_rows": int(train_y.numel()),
        "positive_rows": int(positive),
        "negative_rows": int(negative),
        "balanced_pos_weight": float(pos_weight.item()),
        "final_bce": float(final_loss.detach().item()),
        "identity_gradient_norm": 0.0,
        "head_gradient_norm": float(head_grad_norm),
        "split": split_receipt,
        "episodes": episode_rows,
    }
    return GIEpiORFitResult(runtime=runtime, reference_mask=reference, query_mask=query, receipt=receipt)


def runtime_state_bytes(runtime: GIEpiORRuntime) -> int:
    tensors = list(runtime.parameters()) + list(runtime.buffers())
    return int(sum(tensor.numel() * tensor.element_size() for tensor in tensors))


def bundle_payload(result: GIEpiORFitResult, class_ids: Sequence[str]) -> Mapping[str, Any]:
    head_state = result.runtime.head.state_dict()
    return {
        "schema": "cvs.phase1.gi_epior_bundle.v1",
        "class_ids": list(class_ids),
        "threshold": GI_EPIOR_THRESHOLD,
        "prototypes": result.runtime.prototypes.detach().cpu().numpy(),
        "scales": result.runtime.scales.detach().cpu().numpy(),
        "head_state": {key: value.detach().cpu().numpy() for key, value in head_state.items()},
        "fit_receipt": result.receipt,
        "runtime_state_bytes": runtime_state_bytes(result.runtime),
    }


__all__ = [
    "GI_EPIOR_THRESHOLD",
    "GIEpiORError",
    "GIEpiORFitResult",
    "GIEpiORHead",
    "GIEpiORRuntime",
    "bundle_payload",
    "canonical_physical_ids",
    "deterministic_reference_query_split",
    "fit_class_geometry",
    "fit_gi_epior",
    "geometry_descriptors",
    "runtime_state_bytes",
]
