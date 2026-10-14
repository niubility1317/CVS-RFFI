"""Fixed, nonfitting geometry census of visible-mean physical residuals.

This module reads no data and has no model, parameters, optimizer, or RNG calls.
Dataset legality is an assertion of the caller; this pure function can validate
the fixed visible-source metadata and geometry, but cannot establish provenance.
"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral

import torch
from torch.nn import functional as F


SOURCE_RX = (1, 3, 4, 6, 8)
SOURCE_DAYS = (1, 2, 3)
SOURCE_TX = tuple(range(6))
PACKET_COUNT = 6300
CELL_COUNT = 70
NORMALIZE_EPS = 1e-12
_INTEGER_DTYPES = tuple(
    getattr(torch, name)
    for name in ("uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64")
    if hasattr(torch, name)
)


def _integer_metadata(value, name):
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu" or value.layout != torch.strided:
            raise ValueError(f"{name} must be a dense CPU integer tensor or an integer sequence")
        if value.shape != (PACKET_COUNT,) or value.dtype not in _INTEGER_DTYPES:
            raise ValueError(f"{name} must have {PACKET_COUNT} integer entries")
        result = value.to(dtype=torch.int64)
    else:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ValueError(f"{name} must be an integer sequence")
        if len(value) != PACKET_COUNT or any(
            isinstance(item, bool) or not isinstance(item, Integral) for item in value
        ):
            raise ValueError(f"{name} must have {PACKET_COUNT} integer entries")
        try:
            result = torch.tensor(value, dtype=torch.int64)
        except (TypeError, ValueError, RuntimeError, OverflowError) as error:
            raise ValueError(f"{name} entries must fit int64") from error
    return result


def _physical_ids(value):
    if isinstance(value, torch.Tensor):
        if (value.device.type != "cpu" or value.layout != torch.strided
                or value.shape != (PACKET_COUNT,) or value.dtype not in _INTEGER_DTYPES):
            raise ValueError("physical_ids must be a CPU integer tensor of length 6300 or an integer/string sequence")
        identifiers = value.tolist()
    else:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ValueError("physical_ids must be an integer/string sequence")
        identifiers = list(value)
    if len(identifiers) != PACKET_COUNT or any(
        isinstance(item, bool) or not isinstance(item, (Integral, str)) for item in identifiers
    ):
        raise ValueError("physical_ids must contain 6300 opaque integer/string IDs")
    if len(set(identifiers)) != PACKET_COUNT:
        raise ValueError("physical_ids must be distinct across the fixed L6300 cohort")


def _distribution(values):
    result = {"count": int(values.numel())}
    if values.numel() == 0:
        result.update({key: None for key in (
            "mean", "std", "min", "p05", "p25", "median", "p75", "p95", "max"
        )})
        return result
    quantiles = torch.quantile(values, values.new_tensor([.05, .25, .5, .75, .95]))
    result.update({
        "mean": float(values.mean()), "std": float(values.std(unbiased=False)),
        "min": float(values.min()), "max": float(values.max()),
        **{key: float(value) for key, value in zip(
            ("p05", "p25", "median", "p75", "p95"), quantiles
        )},
    })
    return result


def _relation_summary(cos_positive, cos_negative, gap, valid, selected):
    selected_valid = selected & valid
    positive_stats = _distribution(cos_positive[selected_valid])
    negative_stats = _distribution(cos_negative[selected_valid])
    gap_stats = _distribution(gap[selected_valid])
    count = int(selected.sum())
    valid_count = int(selected_valid.sum())
    ordering_count = int((gap[selected_valid] <= 0).sum())
    return {
        "relation_count": count, "valid_count": valid_count,
        "invalid_count": count - valid_count,
        "cos_positive": positive_stats, "cos_negative": negative_stats, "gap": gap_stats,
        "cos_positive_mean": positive_stats["mean"],
        "cos_negative_mean": negative_stats["mean"], "gap_mean": gap_stats["mean"],
        "ordering_le_zero_count": ordering_count,
        "ordering_le_zero_rate": ordering_count / valid_count if valid_count else None,
    }


def _pair_summary(cosine, valid):
    count = int(cosine.numel())
    valid_count = int(valid.sum())
    return {
        "pair_count": count, "valid_count": valid_count, "invalid_count": count - valid_count,
        "cosine": _distribution(cosine[valid]),
    }


@torch.no_grad()
def measure(vectors, labels, receivers, days, physical_ids, valid, role="L_s"):
    """Census all 126000 fixed source relations without fitting or resampling.

    ``vectors`` is the CPU real floating ``[6300, 24]`` visible-mean physical
    residual, not a learned joint160 feature. Each TX/RX/day cell must contain
    exactly 70 distinct physical packets. The original input order within a cell
    defines its ordinal. An anchor is compared to its same-TX/same-day/same-
    ordinal packet in each other RX and to all five other-TX packets at that
    reference RX/day/ordinal. Relation validity is ONLY the conjunction of the
    three caller-provided visible-valid flags; zero residuals are not excluded.

    ``summary.RX/day/TX`` use the anchor metadata; keys are strings. Returned
    per-relation tensors are detached CPU tensors for local independent reuse.
    No source-selection threshold or margin is applied by this function.
    """
    if role != "L_s":
        raise ValueError("measure accepts only visible labelled source role L_s")
    if not isinstance(vectors, torch.Tensor):
        raise ValueError("vectors must be a CPU real floating tensor of shape [6300, 24]")
    if (vectors.device.type != "cpu" or vectors.layout != torch.strided
            or vectors.shape != (PACKET_COUNT, 24) or not vectors.is_floating_point()):
        raise ValueError("vectors must be a CPU real floating tensor of shape [6300, 24]")
    if not bool(torch.isfinite(vectors).all()):
        raise ValueError("vectors must be finite")
    labels = _integer_metadata(labels, "labels")
    receivers = _integer_metadata(receivers, "receivers")
    days = _integer_metadata(days, "days")
    if not bool(((labels >= 0) & (labels < 6)).all()):
        raise ValueError("labels must be true visible source TX integers 0 through 5")
    if not bool(torch.isin(receivers, torch.tensor(SOURCE_RX)).all()):
        raise ValueError("receivers must belong to source RX1/3/4/6/8")
    if not bool(((days >= 1) & (days <= 3)).all()):
        raise ValueError("days must belong to source day1/2/3")
    _physical_ids(physical_ids)
    if (not isinstance(valid, torch.Tensor) or valid.device.type != "cpu"
            or valid.layout != torch.strided or valid.shape != (PACKET_COUNT,)
            or valid.dtype != torch.bool):
        raise ValueError("valid must be the caller's CPU bool tensor of length 6300")

    table = torch.empty((6, 5, 3, CELL_COUNT), dtype=torch.int64)
    ordinals = torch.empty(PACKET_COUNT, dtype=torch.int64)
    receiver_indices = torch.empty(PACKET_COUNT, dtype=torch.int64)
    cell_counts = []
    for tx in SOURCE_TX:
        for rx_index, rx in enumerate(SOURCE_RX):
            for day_index, day in enumerate(SOURCE_DAYS):
                rows = ((labels == tx) & (receivers == rx) & (days == day)).nonzero().flatten()
                if rows.numel() != CELL_COUNT:
                    raise ValueError(f"source cell TX{tx}/RX{rx}/day{day} must contain exactly 70 packets; got {rows.numel()}")
                table[tx, rx_index, day_index] = rows
                ordinals[rows] = torch.arange(CELL_COUNT)
                receiver_indices[rows] = rx_index
                cell_counts.append({"TX": tx, "RX": rx, "day": day, "count": CELL_COUNT})

    packet_indices = torch.arange(PACKET_COUNT)
    rx_grid = torch.arange(5).expand(PACKET_COUNT, 5)
    reference_rx = rx_grid[rx_grid != receiver_indices[:, None]].reshape(PACKET_COUNT, 4)
    tx_grid = torch.arange(6).expand(PACKET_COUNT, 6)
    other_tx = tx_grid[tx_grid != labels[:, None]].reshape(PACKET_COUNT, 5)
    day_indices = days - 1
    positive = table[labels[:, None], reference_rx, day_indices[:, None], ordinals[:, None]]
    negative = table[
        other_tx[:, None, :], reference_rx[:, :, None],
        day_indices[:, None, None], ordinals[:, None, None],
    ]
    anchor = packet_indices[:, None, None].expand(-1, 4, 5).reshape(-1)
    pos = positive[:, :, None].expand(-1, -1, 5).reshape(-1)
    neg = negative.reshape(-1)
    relations = torch.stack((anchor, pos, neg), dim=1)
    relation_valid = valid[anchor] & valid[pos] & valid[neg]

    residuals = vectors.detach().to(dtype=torch.float64)
    norms = torch.linalg.vector_norm(residuals, dim=1)
    if not bool(torch.isfinite(norms).all()):
        raise ValueError("vector norms must be representable as finite float64 values")
    normalized = F.normalize(residuals, dim=1, eps=NORMALIZE_EPS)
    cos_positive = (normalized[anchor] * normalized[pos]).sum(dim=1)
    cos_negative = (normalized[anchor] * normalized[neg]).sum(dim=1)
    gap = cos_positive - cos_negative
    if not bool(torch.isfinite(gap).all()):
        raise ValueError("normalized relation geometry must be finite")

    selected = torch.ones_like(relation_valid)
    overall = _relation_summary(cos_positive, cos_negative, gap, relation_valid, selected)
    groups = {}
    for name, metadata, values in (
        ("RX", receivers, SOURCE_RX), ("day", days, SOURCE_DAYS), ("TX", labels, SOURCE_TX)
    ):
        groups[name] = {
            str(value): _relation_summary(
                cos_positive, cos_negative, gap, relation_valid, metadata[anchor] == value
            ) for value in values
        }

    cross_anchor = packet_indices[:, None].expand(-1, 4).reshape(-1)
    cross_positive = positive.reshape(-1)
    cross_valid = valid[cross_anchor] & valid[cross_positive]
    cross_cosine = (normalized[cross_anchor] * normalized[cross_positive]).sum(dim=1)
    same_anchor = packet_indices[:, None].expand(-1, 5).reshape(-1)
    same_negative = table[
        other_tx, receiver_indices[:, None], day_indices[:, None], ordinals[:, None]
    ].reshape(-1)
    same_valid = valid[same_anchor] & valid[same_negative]
    same_cosine = (normalized[same_anchor] * normalized[same_negative]).sum(dim=1)
    zero = (residuals == 0).all(dim=1)
    summary = {
        "role": role, "vector_kind": "visible_mean_physical_residual24",
        "input_count": PACKET_COUNT, "visible_valid_packet_count": int(valid.sum()),
        "visible_invalid_packet_count": int((~valid).sum()),
        "cell_count": len(cell_counts), "packets_per_cell": CELL_COUNT, "cells": cell_counts,
        "normalize_eps": NORMALIZE_EPS, "statistics_dtype": "float64",
        "zero_vector_count": int(zero.sum()), "valid_zero_vector_count": int((zero & valid).sum()),
        "norm_le_eps_count": int((norms <= NORMALIZE_EPS).sum()),
        "overall": overall, **groups,
        "cross_RX_same_TX_pairs": _pair_summary(cross_cosine, cross_valid),
        "same_RX_other_TX_pairs": _pair_summary(same_cosine, same_valid),
        "relation_plan": "all 4 other RX x 5 other TX at the same day and original within-cell ordinal",
        "grouping": "RX/day/TX statistics use anchor metadata; invalid endpoint relations are excluded only by visible-valid flags",
        "pair_statistics": "ordered two-endpoint pairs, each counted once per anchor and reference; validity uses only those two endpoints",
        "no_fitted_state": True,
        "boundary": "seen-source-RX physical residual geometry; not learned-model performance, DG, new-TX identification or causal nuisance evidence; caller asserts data provenance",
    }
    return {
        "summary": summary,
        "tensors": {
            "relations": relations, "relation_valid": relation_valid,
            "cos_positive": cos_positive, "cos_negative": cos_negative, "gap": gap,
            "cell_ordinals": ordinals,
            "cross_RX_same_TX_pairs": torch.stack((cross_anchor, cross_positive), dim=1),
            "cross_RX_same_TX_valid": cross_valid, "cross_RX_same_TX_cosine": cross_cosine,
            "same_RX_other_TX_pairs": torch.stack((same_anchor, same_negative), dim=1),
            "same_RX_other_TX_valid": same_valid, "same_RX_other_TX_cosine": same_cosine,
        },
    }
