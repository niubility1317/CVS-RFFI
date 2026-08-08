"""Worst-receiver calibrated nearest-competitor threshold readout.

This is a source-only, no-learning Phase1 readout over a frozen GI-EpiOR
bundle.  It deliberately reuses the bundle's class geometry and its NCT
ratio; it never fits a new representation, geometry, or rejector head.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from cvsrffi.phase1_gi_epior import (
    GI_EPIOR_EPS,
    GIEpiORError,
    GIEpiORRuntime,
    deterministic_reference_query_split,
)


WRC_NCT_ALPHA = 0.02
WRC_NCT_MIN_RX_CALIBRATION_ROWS = 50
WRC_NCT_EPS = GI_EPIOR_EPS


class WRCNCTError(ValueError):
    """Raised when the frozen WRC-NCT source-only contract is violated."""


def _as_strings(values: Sequence[Any]) -> np.ndarray:
    return np.asarray([str(value) for value in values], dtype=object)


def _hash_rank(physical_id: str) -> tuple[bytes, str]:
    value = str(physical_id)
    return hashlib.sha256(value.encode("utf-8")).digest(), value


def deterministic_reference_calibration_eval_split(
    tx_ids: Sequence[Any],
    rx_ids: Sequence[Any],
    physical_ids: Sequence[Any],
    source_tx_ids: Sequence[str],
    *,
    min_calibration_rows_per_rx: int = WRC_NCT_MIN_RX_CALIBRATION_ROWS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Split source physical rows into the frozen GI R50/C25/E25 partition.

    The R/Q50 division is delegated to GI-EpiOR verbatim.  Only its query half
    is further deterministically divided into calibration and evaluation.  The
    function intentionally has no access to target-old or proxy rows.
    """

    tx = _as_strings(tx_ids)
    rx = _as_strings(rx_ids)
    physical = _as_strings(physical_ids)
    if not (tx.size == rx.size == physical.size):
        raise WRCNCTError("source split metadata rows must match")
    if int(min_calibration_rows_per_rx) != WRC_NCT_MIN_RX_CALIBRATION_ROWS:
        raise WRCNCTError("WRC-NCT minimum per-RX calibration count is frozen at 50")
    source = tuple(str(value) for value in source_tx_ids)
    try:
        reference, query, gi_receipt = deterministic_reference_query_split(tx, physical, source)
    except GIEpiORError as error:
        raise WRCNCTError(str(error)) from error

    calibration = np.zeros(tx.size, dtype=bool)
    evaluation = np.zeros(tx.size, dtype=bool)
    per_tx: dict[str, dict[str, int]] = {}
    for class_id in source:
        query_indices = np.flatnonzero(query & (tx == class_id))
        ranked = sorted(query_indices.tolist(), key=lambda index: _hash_rank(str(physical[index])))
        cut = len(ranked) // 2
        calibration_rows = np.asarray(ranked[:cut], dtype=np.int64)
        evaluation_rows = np.asarray(ranked[cut:], dtype=np.int64)
        if calibration_rows.size == 0 or evaluation_rows.size == 0:
            raise WRCNCTError(f"source TX {class_id} produced an empty calibration/evaluation split")
        calibration[calibration_rows] = True
        evaluation[evaluation_rows] = True
        per_tx[class_id] = {
            "reference": int(np.sum(reference & (tx == class_id))),
            "calibration": int(calibration_rows.size),
            "evaluation": int(evaluation_rows.size),
        }

    if bool(np.any(reference & calibration)) or bool(np.any(reference & evaluation)) or bool(np.any(calibration & evaluation)):
        raise WRCNCTError("R/C/E row overlap")
    source_mask = np.isin(tx, np.asarray(source, dtype=object))
    if not np.array_equal(reference | calibration | evaluation, source_mask):
        raise WRCNCTError("source rows are not closed by the R/C/E split")
    physical_sets = [set(physical[mask].tolist()) for mask in (reference, calibration, evaluation)]
    if physical_sets[0] & physical_sets[1] or physical_sets[0] & physical_sets[2] or physical_sets[1] & physical_sets[2]:
        raise WRCNCTError("R/C/E physical ID overlap")

    per_rx: dict[str, int] = {}
    for receiver_id in sorted(set(rx[source_mask].tolist())):
        count = int(np.sum(calibration & (rx == receiver_id)))
        if count < WRC_NCT_MIN_RX_CALIBRATION_ROWS:
            raise WRCNCTError(
                f"source RX {receiver_id} has {count} calibration rows; requires at least "
                f"{WRC_NCT_MIN_RX_CALIBRATION_ROWS}"
            )
        per_rx[receiver_id] = count
    if not per_rx:
        raise WRCNCTError("source split contains no receiver IDs")
    receipt = {
        "schema": "cvs.phase1.wrc_nct_split.v1",
        "source_tx_ids": list(source),
        "reference_rows": int(reference.sum()),
        "calibration_rows": int(calibration.sum()),
        "evaluation_rows": int(evaluation.sum()),
        "physical_overlap": 0,
        "per_tx": per_tx,
        "calibration_rows_per_rx": per_rx,
        "gi_reference_query_receipt": gi_receipt,
    }
    return reference, calibration, evaluation, receipt


def finite_upper_quantile(
    scores: Sequence[float] | np.ndarray,
    *,
    alpha: float = WRC_NCT_ALPHA,
) -> tuple[float, int, int]:
    """Return the prescribed finite-sample upper quantile and its 1-based rank."""

    if float(alpha) != WRC_NCT_ALPHA:
        raise WRCNCTError("WRC-NCT alpha is frozen at 0.02")
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise WRCNCTError("calibration NCT scores must be non-empty and finite")
    ordered = np.sort(values)
    n = int(ordered.size)
    k = min(n, int(math.ceil((1.0 - WRC_NCT_ALPHA) * float(n + 1))))
    if k < 1:
        raise WRCNCTError("finite quantile rank must be positive")
    return float(ordered[k - 1]), int(k), n


class WRCNCTRuntime(nn.Module):
    """Small deployable threshold runtime with no GI-EpiOR rejector head."""

    def __init__(
        self,
        prototypes: torch.Tensor,
        scales: torch.Tensor,
        tau: float,
        *,
        eps: float = WRC_NCT_EPS,
    ) -> None:
        super().__init__()
        prototype = torch.as_tensor(prototypes, dtype=torch.float32).detach()
        rho = torch.as_tensor(scales, dtype=torch.float32).detach().reshape(-1)
        if prototype.ndim != 2 or prototype.size(0) < 2:
            raise WRCNCTError("GI class geometry must contain at least two prototypes")
        if rho.numel() != prototype.size(0) or not bool(torch.isfinite(rho).all()) or bool(torch.any(rho <= 0.0)):
            raise WRCNCTError("GI MAD scales must be finite and positive")
        if not math.isfinite(float(tau)):
            raise WRCNCTError("WRC-NCT threshold must be finite")
        norms = torch.linalg.vector_norm(prototype, dim=1)
        if not bool(torch.allclose(norms, torch.ones_like(norms), atol=1.0e-5, rtol=1.0e-5)):
            raise WRCNCTError("upstream GI prototypes must already be unit normalized")
        # Preserve the upstream runtime geometry byte-for-byte.  Re-normalizing
        # an already normalized float32 tensor can move the last bit; the very
        # small frozen MAD scales may then amplify that into a parity failure.
        self.register_buffer("prototypes", prototype)
        self.register_buffer("scales", rho)
        self.register_buffer("tau", torch.tensor(float(tau), dtype=torch.float32))
        self.eps = float(eps)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z = F.normalize(features.float(), dim=1)
        d_class = (1.0 - z @ self.prototypes.T) / (self.scales.unsqueeze(0) + self.eps)
        ordered = torch.sort(d_class, dim=1).values
        d1 = ordered[:, 0]
        d2 = ordered[:, 1]
        ratio = d1 / (d2 + self.eps)
        accepted = ratio <= self.tau
        return d1, d2, ratio, accepted


@dataclass
class WRCNCTFitResult:
    runtime: WRCNCTRuntime
    reference_mask: np.ndarray
    calibration_mask: np.ndarray
    evaluation_mask: np.ndarray
    receipt: dict[str, Any]


def fit_wrc_nct(
    features: torch.Tensor,
    tx_ids: Sequence[Any],
    rx_ids: Sequence[Any],
    physical_ids: Sequence[Any],
    source_tx_ids: Sequence[str],
    gi_runtime: GIEpiORRuntime,
    *,
    alpha: float = WRC_NCT_ALPHA,
) -> WRCNCTFitResult:
    """Calibrate one WRC-NCT threshold from source C25 only."""

    if float(alpha) != WRC_NCT_ALPHA:
        raise WRCNCTError("WRC-NCT alpha is frozen at 0.02")
    reference, calibration, evaluation, split_receipt = deterministic_reference_calibration_eval_split(
        tx_ids, rx_ids, physical_ids, source_tx_ids
    )
    source_features = torch.as_tensor(features, dtype=torch.float32)
    if source_features.ndim != 2 or source_features.size(0) != calibration.size:
        raise WRCNCTError("source features must match split metadata")
    if not bool(torch.isfinite(source_features).all()):
        raise WRCNCTError("source features must be finite")
    with torch.no_grad():
        _, _, ratio_t = gi_runtime.eval()(source_features)
    ratio = np.asarray(ratio_t.detach().cpu().tolist(), dtype=np.float64).reshape(-1)
    if ratio.size != calibration.size or not np.isfinite(ratio).all():
        raise WRCNCTError("GI runtime NCT ratio is invalid")
    rx = _as_strings(rx_ids)
    tau_r: dict[str, float] = {}
    rank_r: dict[str, dict[str, int]] = {}
    for receiver_id in sorted(set(rx[calibration].tolist())):
        values = ratio[calibration & (rx == receiver_id)]
        if values.size < WRC_NCT_MIN_RX_CALIBRATION_ROWS:
            raise WRCNCTError(f"source RX {receiver_id} violates the frozen C>=50 contract")
        tau, k, n = finite_upper_quantile(values, alpha=alpha)
        tau_r[receiver_id] = tau
        rank_r[receiver_id] = {"k": k, "n": n}
    if not tau_r:
        raise WRCNCTError("WRC-NCT calibration has no source receivers")
    tau = float(max(tau_r.values()))
    runtime = WRCNCTRuntime(gi_runtime.prototypes, gi_runtime.scales, tau, eps=gi_runtime.eps).eval()
    receipt = {
        "schema": "cvs.phase1.wrc_nct_readout.v1",
        "method": "WRC-NCT",
        "alpha": WRC_NCT_ALPHA,
        "score_formula": "d1/(d2+eps)_from_frozen_gi_runtime",
        "threshold_policy": "tau=max_source_rx_finite_q0p98_no_outer_calibration",
        "tau": tau,
        "tau_r": tau_r,
        "finite_quantile_rank": rank_r,
        "split": split_receipt,
        "learning_head": False,
        "new_geometry": False,
        "outer_rows_used_for_fit_or_calibration": 0,
    }
    return WRCNCTFitResult(
        runtime=runtime,
        reference_mask=reference,
        calibration_mask=calibration,
        evaluation_mask=evaluation,
        receipt=receipt,
    )


def runtime_state_bytes(runtime: WRCNCTRuntime) -> int:
    tensors = list(runtime.parameters()) + list(runtime.buffers())
    return int(sum(tensor.numel() * tensor.element_size() for tensor in tensors))


__all__ = [
    "WRC_NCT_ALPHA",
    "WRC_NCT_EPS",
    "WRC_NCT_MIN_RX_CALIBRATION_ROWS",
    "WRCNCTError",
    "WRCNCTFitResult",
    "WRCNCTRuntime",
    "deterministic_reference_calibration_eval_split",
    "finite_upper_quantile",
    "fit_wrc_nct",
    "runtime_state_bytes",
]
