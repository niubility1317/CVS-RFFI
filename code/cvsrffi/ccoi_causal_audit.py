"""Pure, source-only diagnostics for the CCOI-PA-V2 causal audit."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, Mapping

import torch
from torch import Tensor
import torch.nn.functional as F


def normalized_energy_fit_score(nmse: float) -> float:
    """Return ``1-NMSE`` without claiming mean-centered coefficient of determination."""

    value = float(nmse)
    if not math.isfinite(value):
        raise ValueError("nmse must be finite")
    return 1.0 - value


def _finite_tensor(value: Tensor, name: str) -> Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return value


def token_code_audit(code_prob: Tensor) -> dict[str, Any]:
    """Separate token-hard assignments from packet-dominant assignments."""

    probabilities = _finite_tensor(code_prob, "code_prob").detach().float().cpu()
    if probabilities.ndim != 3 or probabilities.size(-1) < 2:
        raise ValueError("code_prob must have shape [packet,token,code] with at least two codes")
    if bool((probabilities < 0).any().item()):
        raise ValueError("code_prob must be non-negative")
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    packet_count, token_count, code_count = probabilities.shape
    token_hard = probabilities.argmax(dim=-1)
    packet_dominant = probabilities.mean(dim=1).argmax(dim=-1)
    token_hist = torch.bincount(token_hard.reshape(-1), minlength=code_count)
    packet_hist = torch.bincount(packet_dominant, minlength=code_count)
    position_hist = [
        torch.bincount(token_hard[:, position], minlength=code_count).tolist()
        for position in range(token_count)
    ]
    transitions = torch.zeros((code_count, code_count), dtype=torch.long)
    if token_count > 1:
        left = token_hard[:, :-1].reshape(-1)
        right = token_hard[:, 1:].reshape(-1)
        flat = left * code_count + right
        transitions += torch.bincount(flat, minlength=code_count * code_count).reshape(code_count, code_count)
    codes_per_packet = torch.tensor(
        [int(torch.unique(row).numel()) for row in token_hard], dtype=torch.float32
    )
    soft_mass = probabilities.mean(dim=(0, 1)).clamp_min(1e-12)
    soft_mass = soft_mass / soft_mass.sum()
    soft_entropy = float(-(soft_mass * soft_mass.log()).sum().item())
    return {
        "packet_count": int(packet_count),
        "token_count_per_packet": int(token_count),
        "code_count": int(code_count),
        "token_hard_histogram": token_hist.tolist(),
        "token_hard_observed": int((token_hist > 0).sum().item()),
        "packet_dominant_histogram": packet_hist.tolist(),
        "packet_dominant_observed": int((packet_hist > 0).sum().item()),
        "codes_per_packet_mean": float(codes_per_packet.mean().item()),
        "codes_per_packet_min": int(codes_per_packet.min().item()),
        "codes_per_packet_max": int(codes_per_packet.max().item()),
        "position_hard_histogram": position_hist,
        "transition_matrix": transitions.tolist(),
        "soft_effective_codes": float(math.exp(soft_entropy)),
        "soft_max_probability": float(soft_mass.max().item()),
        "soft_entropy": soft_entropy,
    }


def _summary_q(q: Tensor) -> Tensor:
    q = _finite_tensor(q, "q").detach().float().cpu()
    if q.ndim == 3:
        q = q.mean(dim=1)
    if q.ndim != 2:
        raise ValueError("q must have shape [sample,dim] or [sample,token,dim]")
    return F.normalize(q, dim=1, eps=1e-8)


def pair_relation_counts(
    q: Tensor,
    tx: Tensor,
    receiver: Tensor,
    min_cosine: float,
    *,
    chunk_size: int = 2048,
) -> dict[str, Any]:
    """Count complete-set challenge relations without depending on loader batches."""

    return pair_relation_sweep(
        q,
        tx,
        receiver,
        thresholds=(float(min_cosine),),
        chunk_size=chunk_size,
    )[f"{float(min_cosine):.3f}"]


def pair_relation_sweep(
    q: Tensor,
    tx: Tensor,
    receiver: Tensor,
    thresholds: Iterable[float],
    *,
    chunk_size: int = 2048,
) -> dict[str, dict[str, Any]]:
    """Evaluate multiple global cosine thresholds in one chunked matrix pass."""

    features = _summary_q(q)
    tx = _finite_tensor(tx, "tx").detach().view(-1).long().cpu()
    receiver = _finite_tensor(receiver, "receiver").detach().view(-1).long().cpu()
    count = int(features.size(0))
    if tx.numel() != count or receiver.numel() != count:
        raise ValueError("q, tx and receiver must contain the same number of samples")
    if count < 2:
        raise ValueError("at least two samples are required")
    threshold_values = tuple(float(value) for value in thresholds)
    if not threshold_values or any(not -1.0 <= value <= 1.0 for value in threshold_values):
        raise ValueError("thresholds must be a non-empty collection within [-1,1]")
    chunk_size = max(1, int(chunk_size))
    results = {
        value: {
            "same_tx_cross_rx_matched": 0,
            "same_tx_cross_rx_total": 0,
            "cross_tx_same_rx_matched": 0,
            "cross_tx_same_rx_total": 0,
            "positive_anchor": torch.zeros(count, dtype=torch.bool),
            "negative_anchor": torch.zeros(count, dtype=torch.bool),
        }
        for value in threshold_values
    }
    indices = torch.arange(count)
    for start in range(0, count, chunk_size):
        stop = min(count, start + chunk_size)
        cosine = features[start:stop] @ features.T
        left_tx = tx[start:stop, None]
        left_receiver = receiver[start:stop, None]
        upper = indices[None, :] > indices[start:stop, None]
        positive = upper & left_tx.eq(tx[None, :]) & left_receiver.ne(receiver[None, :])
        negative = upper & left_tx.ne(tx[None, :]) & left_receiver.eq(receiver[None, :])
        positive_total = int(positive.sum().item())
        negative_total = int(negative.sum().item())
        for threshold in threshold_values:
            matched = cosine >= threshold
            positive_matched = positive & matched
            negative_matched = negative & matched
            result = results[threshold]
            result["same_tx_cross_rx_total"] += positive_total
            result["same_tx_cross_rx_matched"] += int(positive_matched.sum().item())
            result["cross_tx_same_rx_total"] += negative_total
            result["cross_tx_same_rx_matched"] += int(negative_matched.sum().item())
            for mask, anchor_name in (
                (positive_matched, "positive_anchor"),
                (negative_matched, "negative_anchor"),
            ):
                local_rows, columns = torch.nonzero(mask, as_tuple=True)
                if local_rows.numel():
                    result[anchor_name][start + local_rows] = True
                    result[anchor_name][columns] = True
    finalized: dict[str, dict[str, Any]] = {}
    for threshold in threshold_values:
        result = results[threshold]
        positive_total = int(result["same_tx_cross_rx_total"])
        negative_total = int(result["cross_tx_same_rx_total"])
        positive_anchor = result.pop("positive_anchor")
        negative_anchor = result.pop("negative_anchor")
        result.update(
            {
                "min_cosine": threshold,
                "sample_count": count,
                "positive_match_rate": result["same_tx_cross_rx_matched"] / max(1, positive_total),
                "negative_match_rate": result["cross_tx_same_rx_matched"] / max(1, negative_total),
                "positive_anchor_coverage": float(positive_anchor.float().mean().item()),
                "negative_anchor_coverage": float(negative_anchor.float().mean().item()),
            }
        )
        finalized[f"{threshold:.3f}"] = result
    return finalized


def build_factor_indices(
    tx: Tensor,
    receiver: Tensor,
    day: Tensor,
    seed: int,
    *,
    q: Tensor | None = None,
) -> dict[str, Tensor]:
    """Build deterministic support substitutions for H2-H6 factorization rows."""

    tx = _finite_tensor(tx, "tx").detach().view(-1).long().cpu()
    receiver = _finite_tensor(receiver, "receiver").detach().view(-1).long().cpu()
    day = _finite_tensor(day, "day").detach().view(-1).long().cpu()
    count = int(tx.numel())
    if count < 2 or receiver.numel() != count or day.numel() != count:
        raise ValueError("tx, receiver and day must have the same length of at least two")
    generator = torch.Generator().manual_seed(int(seed))
    q_summary = _summary_q(q) if q is not None else None
    if q_summary is not None and q_summary.size(0) != count:
        raise ValueError("q must align with tx, receiver and day")
    order = torch.randperm(count, generator=generator)
    shuffled = torch.empty(count, dtype=torch.long)
    shuffled[order] = order.roll(-1)
    same_rx_day: dict[tuple[int, int], list[int]] = defaultdict(list)
    same_rx: dict[int, list[int]] = defaultdict(list)
    same_tx: dict[int, list[int]] = defaultdict(list)
    for index in range(count):
        same_rx_day[(int(receiver[index]), int(day[index]))].append(index)
        same_rx[int(receiver[index])].append(index)
        same_tx[int(tx[index])].append(index)

    def choose(index: int, candidates: list[int]) -> int:
        if not candidates:
            return -1
        if q_summary is not None:
            candidate_tensor = torch.tensor(candidates, dtype=torch.long)
            similarity = q_summary[candidate_tensor] @ q_summary[index]
            return int(candidate_tensor[int(similarity.argmax().item())])
        position = int(torch.randint(len(candidates), (1,), generator=generator).item())
        return int(candidates[position])

    h4 = torch.full((count,), -1, dtype=torch.long)
    h5 = torch.full((count,), -1, dtype=torch.long)
    h6 = torch.full((count,), -1, dtype=torch.long)
    all_indices = list(range(count))
    for index in range(count):
        tx_i, rx_i, day_i = int(tx[index]), int(receiver[index]), int(day[index])
        other_tx = [j for j in same_rx_day[(rx_i, day_i)] if int(tx[j]) != tx_i]
        if not other_tx:
            other_tx = [j for j in same_rx[rx_i] if int(tx[j]) != tx_i]
        if not other_tx:
            other_tx = [j for j in all_indices if int(tx[j]) != tx_i]
        h4[index] = choose(index, other_tx)
        cross_rx = [j for j in same_tx[tx_i] if int(receiver[j]) != rx_i and int(day[j]) == day_i]
        if not cross_rx:
            cross_rx = [j for j in same_tx[tx_i] if int(receiver[j]) != rx_i]
        h5[index] = choose(index, cross_rx)
        cross_day = [j for j in same_tx[tx_i] if int(day[j]) != day_i and int(receiver[j]) == rx_i]
        if not cross_day:
            cross_day = [j for j in same_tx[tx_i] if int(day[j]) != day_i]
        h6[index] = choose(index, cross_day)
    identity = torch.arange(count)
    return {"H2": identity, "H3": shuffled, "H4": h4, "H5": h5, "H6": h6}


def group_paired_bootstrap(
    reference_error: Tensor,
    candidate_error: Tensor,
    groups: Tensor,
    *,
    resamples: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Paired cluster bootstrap of relative error reduction."""

    reference = _finite_tensor(reference_error, "reference_error").detach().view(-1).double().cpu()
    candidate = _finite_tensor(candidate_error, "candidate_error").detach().view(-1).double().cpu()
    groups = _finite_tensor(groups, "groups").detach().cpu()
    if groups.ndim == 1:
        groups = groups[:, None]
    if reference.numel() == 0 or candidate.numel() != reference.numel() or groups.size(0) != reference.numel():
        raise ValueError("errors and groups must have the same non-zero sample count")
    if bool((reference < 0).any().item()) or bool((candidate < 0).any().item()):
        raise ValueError("squared errors must be non-negative")
    _, inverse = torch.unique(groups, dim=0, return_inverse=True)
    group_count = int(inverse.max().item()) + 1
    ref_sum = torch.zeros(group_count, dtype=torch.double).scatter_add_(0, inverse, reference)
    cand_sum = torch.zeros(group_count, dtype=torch.double).scatter_add_(0, inverse, candidate)
    sizes = torch.zeros(group_count, dtype=torch.double).scatter_add_(0, inverse, torch.ones_like(reference))

    def gain(selected: Tensor) -> Tensor:
        ref_mean = ref_sum[selected].sum() / sizes[selected].sum().clamp_min(1.0)
        cand_mean = cand_sum[selected].sum() / sizes[selected].sum().clamp_min(1.0)
        return (ref_mean - cand_mean) / ref_mean.clamp_min(1e-12)

    point = float(gain(torch.arange(group_count)).item())
    generator = torch.Generator().manual_seed(int(seed))
    draws = []
    for _ in range(max(1, int(resamples))):
        selected = torch.randint(group_count, (group_count,), generator=generator)
        draws.append(gain(selected))
    distribution = torch.stack(draws)
    return {
        "sample_count": int(reference.numel()),
        "group_count": group_count,
        "resamples": max(1, int(resamples)),
        "relative_gain": point,
        "ci95_low": float(torch.quantile(distribution, 0.025).item()),
        "ci95_high": float(torch.quantile(distribution, 0.975).item()),
    }


def complementarity_table(
    base_prediction: Tensor,
    side_prediction: Tensor,
    truth: Tensor,
) -> dict[str, Any]:
    base = _finite_tensor(base_prediction, "base_prediction").detach().view(-1).long().cpu()
    side = _finite_tensor(side_prediction, "side_prediction").detach().view(-1).long().cpu()
    labels = _finite_tensor(truth, "truth").detach().view(-1).long().cpu()
    if not (base.numel() == side.numel() == labels.numel()):
        raise ValueError("base, side and truth must contain the same number of samples")
    if labels.numel() == 0:
        raise ValueError("at least one prediction is required")
    base_ok = base.eq(labels)
    side_ok = side.eq(labels)
    both_correct = int((base_ok & side_ok).sum().item())
    rescue = int((~base_ok & side_ok).sum().item())
    harm = int((base_ok & ~side_ok).sum().item())
    both_wrong = int((~base_ok & ~side_ok).sum().item())
    total = int(labels.numel())
    return {
        "sample_count": total,
        "both_correct": both_correct,
        "base_wrong_side_correct": rescue,
        "base_correct_side_wrong": harm,
        "both_wrong": both_wrong,
        "base_accuracy": float(base_ok.float().mean().item()),
        "side_accuracy": float(side_ok.float().mean().item()),
        "oracle_accuracy": float((base_ok | side_ok).float().mean().item()),
        "rescue_minus_harm": rescue - harm,
    }


def factorized_holdout_metrics(
    predictions: Mapping[str, Tensor],
    target: Tensor,
) -> dict[str, Any]:
    target = _finite_tensor(target, "target").detach().float()
    if target.numel() == 0:
        raise ValueError("target must not be empty")
    energy = float(target.square().sum().item())
    if energy <= 0.0:
        raise ValueError("target energy must be positive")
    rows: dict[str, Any] = {}
    for name, prediction in predictions.items():
        prediction = _finite_tensor(prediction, f"prediction[{name}]").detach().float()
        if prediction.shape != target.shape:
            raise ValueError(f"prediction[{name}] must match target shape")
        squared_error = float((prediction - target).square().sum().item())
        nmse = squared_error / energy
        rows[str(name)] = {
            "squared_error": squared_error,
            "nmse": nmse,
            "normalized_energy_fit_score": normalized_energy_fit_score(nmse),
        }
    return {"sample_count": int(target.shape[0]), "target_energy": energy, "rows": rows}


__all__ = [
    "build_factor_indices",
    "complementarity_table",
    "factorized_holdout_metrics",
    "group_paired_bootstrap",
    "normalized_energy_fit_score",
    "pair_relation_counts",
    "pair_relation_sweep",
    "token_code_audit",
]
