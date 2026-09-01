"""Support-only MARC-OT objectives and blockwise gradient projection."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as functional
from torch import Tensor

from .meta_weight_bank import parameter_block_key
from .stage2_wiser_p3 import (
    cross_fitted_p3_loss,
    frozen_class_risk,
    stratified_crossfit_indices,
)
from .stage2_wiser_rf import leave_one_out_prototype_logits


_MARGINAL_TOLERANCE = 1.0e-4


@dataclass(frozen=True)
class MARCOTDiagnostics:
    """Differentiable support losses plus detached calibration diagnostics.

    Temporary fold prototypes and transported bank rows are deliberately local
    variables in :func:`marc_ot_losses`; neither is part of this returned state.
    """

    total: Tensor
    frozen_head_ce: Tensor
    cross_fit_ce: Tensor
    leave_one_out_ce: Tensor
    class_risk: Tensor
    class_risk_loss: Tensor
    transport_loss: Tensor
    statistics_loss: Tensor
    k_shot: int
    statistics_mode: str
    cross_fit_mode: str
    transport_row_error: float
    transport_column_error: float


class _ProjectedGradientMap(dict[str, list[Tensor | None]]):
    """Dictionary-compatible projected gradients with per-block diagnostics."""

    def __init__(
        self,
        values: Mapping[str, list[Tensor | None]],
        diagnostics: Mapping[str, Mapping[str, Any]],
    ) -> None:
        super().__init__(values)
        self.diagnostics = {name: dict(row) for name, row in diagnostics.items()}


def _finite_positive_scalar(value: float, *, name: str) -> float:
    scalar = float(value)
    if not math.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return scalar


def support_bank_transport(
    support_task_features: Tensor,
    frozen_bank_task_features: Tensor,
    epsilon: float,
    iterations: int,
) -> Tensor:
    """Return a uniform support-to-frozen-bank FP32 log-Sinkhorn plan."""

    if not isinstance(support_task_features, Tensor) or not isinstance(
        frozen_bank_task_features, Tensor
    ):
        raise ValueError("support and bank task features must be torch tensors")
    support = support_task_features
    bank = frozen_bank_task_features
    if support.ndim != 2 or bank.ndim != 2:
        raise ValueError("support and bank task features must be two-dimensional")
    if support.shape[0] < 1 or bank.shape[0] < 1:
        raise ValueError("support and bank task features must be nonempty")
    if support.shape[1] != bank.shape[1]:
        raise ValueError("support and bank feature dimensions must match")
    if not support.is_floating_point() or not bank.is_floating_point():
        raise ValueError("support and bank task features must be floating point")
    if not bool(torch.isfinite(support).all()) or not bool(torch.isfinite(bank).all()):
        raise ValueError("support and bank task features must be finite")
    epsilon_value = _finite_positive_scalar(epsilon, name="epsilon")
    if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations < 1:
        raise ValueError("iterations must be a positive integer")

    support_work = support.float()
    bank_work = bank.detach().to(device=support.device, dtype=torch.float32)
    cost = torch.cdist(support_work, bank_work).square()
    if not bool(torch.isfinite(cost).all()):
        raise ValueError("support-bank transport cost became nonfinite")
    log_a = cost.new_full((cost.shape[0],), -math.log(cost.shape[0]))
    log_b = cost.new_full((cost.shape[1],), -math.log(cost.shape[1]))
    log_k = -cost / epsilon_value
    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)
    for _step in range(iterations):
        log_u = log_a - torch.logsumexp(log_k + log_v[None, :], dim=1)
        log_v = log_b - torch.logsumexp(log_k + log_u[:, None], dim=0)
    plan = torch.exp(log_u[:, None] + log_k + log_v[None, :])
    if not bool(torch.isfinite(plan).all()):
        raise ValueError("support-bank transport became nonfinite")

    with torch.no_grad():
        row_target = plan.new_full((plan.shape[0],), 1.0 / float(plan.shape[0]))
        column_target = plan.new_full((plan.shape[1],), 1.0 / float(plan.shape[1]))
        row_error = float((plan.sum(dim=1) - row_target).abs().max())
        column_error = float((plan.sum(dim=0) - column_target).abs().max())
    if (
        not math.isfinite(row_error)
        or not math.isfinite(column_error)
        or row_error > _MARGINAL_TOLERANCE
        or column_error > _MARGINAL_TOLERANCE
    ):
        raise ValueError(
            "support-bank transport marginals failed to converge: "
            f"row_error={row_error:.6g} column_error={column_error:.6g}"
        )
    return plan


def _block_name(name: str) -> str:
    value = str(name)
    if not value:
        raise ValueError("gradient block names must be nonempty")
    routed = parameter_block_key(value)
    if routed is not None:
        return routed
    if "." in value:
        raise ValueError(f"parameter name is outside the canonical block routing: {value!r}")
    return value


def _finite_from_logs(sign: float, log_abs: float) -> float:
    if sign == 0.0:
        return 0.0
    limit = torch.finfo(torch.float64).max
    if log_abs >= math.log(limit):
        return math.copysign(float(limit), sign)
    return math.copysign(math.exp(log_abs), sign)


def _scaled_norm(scale: Tensor, normalized_square: Tensor) -> float:
    if bool(scale == 0.0) or bool(normalized_square == 0.0):
        return 0.0
    return _finite_from_logs(
        1.0,
        math.log(float(scale)) + 0.5 * math.log(float(normalized_square)),
    )


def _scaled_dot(primary_scale: Tensor, auxiliary_scale: Tensor, dot: Tensor) -> float:
    value = float(dot)
    if value == 0.0 or bool(primary_scale == 0.0) or bool(auxiliary_scale == 0.0):
        return 0.0
    return _finite_from_logs(
        value,
        math.log(float(primary_scale))
        + math.log(float(auxiliary_scale))
        + math.log(abs(value)),
    )


def _projection_factor(
    primary_scale: Tensor,
    auxiliary_scale: Tensor,
    primary_square: Tensor,
    projected_square: Tensor,
    ratio_cap: float,
) -> float:
    if ratio_cap == 0.0 or bool(primary_square == 0.0):
        return 0.0
    if bool(projected_square == 0.0) or bool(auxiliary_scale == 0.0):
        return 1.0
    log_ratio = (
        math.log(ratio_cap)
        + math.log(float(primary_scale))
        + 0.5 * math.log(float(primary_square))
        - math.log(float(auxiliary_scale))
        - 0.5 * math.log(float(projected_square))
    )
    return 1.0 if log_ratio >= 0.0 else math.exp(log_ratio)


def _epsilon_in_scaled_primary_units(eps: float, primary_scale: Tensor) -> float:
    if bool(primary_scale == 0.0):
        return float(torch.finfo(torch.float64).max)
    log_value = math.log(eps) - 2.0 * math.log(float(primary_scale))
    if log_value >= math.log(float(torch.finfo(torch.float64).max)):
        return float(torch.finfo(torch.float64).max)
    if log_value <= math.log(float(torch.finfo(torch.float64).tiny)):
        return 0.0
    return math.exp(log_value)


def blockwise_primary_projection(
    primary: Mapping[str, Sequence[Tensor | None]],
    auxiliary: Mapping[str, Sequence[Tensor | None]],
    *,
    ratio_cap: float,
    eps: float = 1.0e-12,
) -> Mapping[str, list[Tensor | None]]:
    """Project calibration gradients per canonical parameter block.

    Dotted parameter names are grouped through the single Task1
    :func:`parameter_block_key` router.  Plain keys are treated as already
    grouped diagnostic names, which keeps the low-level helper usable with the
    hand-specified blocks in the implementation brief.
    """

    if not isinstance(primary, Mapping) or not isinstance(auxiliary, Mapping):
        raise ValueError("primary and auxiliary gradients must be mappings")
    if set(primary) != set(auxiliary):
        raise ValueError("primary and auxiliary gradients must have identical keys")
    cap = float(ratio_cap)
    if not math.isfinite(cap) or cap < 0.0:
        raise ValueError("ratio_cap must be finite and nonnegative")
    epsilon_value = _finite_positive_scalar(eps, name="eps")

    grouped: dict[str, list[tuple[Tensor | None, Tensor | None]]] = {}
    for raw_name in primary:
        primary_slots = primary[raw_name]
        auxiliary_slots = auxiliary[raw_name]
        if (
            not isinstance(primary_slots, Sequence)
            or isinstance(primary_slots, Tensor)
            or not isinstance(auxiliary_slots, Sequence)
            or isinstance(auxiliary_slots, Tensor)
        ):
            raise ValueError("each gradient block must contain a sequence of slots")
        if len(primary_slots) != len(auxiliary_slots):
            raise ValueError("corresponding gradient blocks must have equal slot counts")
        block = _block_name(raw_name)
        grouped.setdefault(block, []).extend(zip(primary_slots, auxiliary_slots))
    if not grouped:
        raise ValueError("at least one gradient block is required")

    projected_by_block: dict[str, list[Tensor | None]] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    for block, raw_slots in grouped.items():
        resolved: list[tuple[Tensor, Tensor, torch.dtype]] = []
        output_positions: list[int] = []
        result_slots: list[Tensor | None] = [None] * len(raw_slots)
        block_device: torch.device | None = None
        for index, (primary_item, auxiliary_item) in enumerate(raw_slots):
            if primary_item is None and auxiliary_item is None:
                continue
            if primary_item is not None and not isinstance(primary_item, Tensor):
                raise TypeError("primary gradient slots must be tensors or None")
            if auxiliary_item is not None and not isinstance(auxiliary_item, Tensor):
                raise TypeError("auxiliary gradient slots must be tensors or None")
            reference = primary_item if primary_item is not None else auxiliary_item
            assert reference is not None
            if not reference.is_floating_point():
                raise TypeError("gradient slots must use floating-point tensors")
            if primary_item is not None and auxiliary_item is not None and (
                primary_item.shape != auxiliary_item.shape
                or primary_item.device != auxiliary_item.device
                or primary_item.dtype != auxiliary_item.dtype
            ):
                raise ValueError("corresponding gradient slots must match shape, device, and dtype")
            if block_device is None:
                block_device = reference.device
            elif reference.device != block_device:
                raise ValueError("all gradient slots in one block must share a device")
            primary_value = torch.zeros_like(reference) if primary_item is None else primary_item
            auxiliary_value = (
                torch.zeros_like(reference) if auxiliary_item is None else auxiliary_item
            )
            if not bool(torch.isfinite(primary_value).all()):
                raise ValueError(f"primary gradient contains nonfinite values in block {block!r}")
            if not bool(torch.isfinite(auxiliary_value).all()):
                raise ValueError(f"auxiliary gradient contains nonfinite values in block {block!r}")
            resolved.append((primary_value, auxiliary_value, reference.dtype))
            output_positions.append(index)

        if not resolved:
            projected_by_block[block] = result_slots
            diagnostics[block] = {
                "raw_cosine": None,
                "projected_cosine": None,
                "primary_norm": 0.0,
                "auxiliary_norm": 0.0,
                "projected_norm": 0.0,
                "raw_dot": 0.0,
                "projected_dot": 0.0,
                "conflict": False,
            }
            continue

        primary_scale = torch.stack(
            [value.detach().abs().to(torch.float64).amax() for value, _, _ in resolved]
        ).amax()
        auxiliary_scale = torch.stack(
            [value.detach().abs().to(torch.float64).amax() for _, value, _ in resolved]
        ).amax()
        if bool(primary_scale == 0.0):
            primary_scaled = tuple(value.to(torch.float64) for value, _, _ in resolved)
        else:
            primary_scaled = tuple(
                value.to(torch.float64) / primary_scale for value, _, _ in resolved
            )
        if bool(auxiliary_scale == 0.0):
            auxiliary_scaled = tuple(value.to(torch.float64) for _, value, _ in resolved)
        else:
            auxiliary_scaled = tuple(
                value.to(torch.float64) / auxiliary_scale for _, value, _ in resolved
            )
        primary_square = torch.stack([value.square().sum() for value in primary_scaled]).sum()
        auxiliary_square = torch.stack([value.square().sum() for value in auxiliary_scaled]).sum()
        raw_dot_scaled = torch.stack(
            [
                (primary_value * auxiliary_value).sum()
                for primary_value, auxiliary_value in zip(primary_scaled, auxiliary_scaled)
            ]
        ).sum()
        conflict = bool(raw_dot_scaled < 0.0) and bool(primary_square > 0.0)
        projected_scaled = auxiliary_scaled
        if conflict:
            scaled_epsilon = _epsilon_in_scaled_primary_units(
                epsilon_value, primary_scale
            )
            coefficient = raw_dot_scaled / (primary_square + scaled_epsilon)
            projected_scaled = tuple(
                auxiliary_value - coefficient * primary_value
                for primary_value, auxiliary_value in zip(primary_scaled, auxiliary_scaled)
            )
            residual = torch.stack(
                [
                    (primary_value * projected_value).sum()
                    for primary_value, projected_value in zip(primary_scaled, projected_scaled)
                ]
            ).sum()
            if bool(residual < 0.0) and abs(float(residual)) <= 1.0e-10:
                projected_scaled = tuple(
                    projected_value - (residual / primary_square) * primary_value
                    for primary_value, projected_value in zip(primary_scaled, projected_scaled)
                )
        projected_square_before_cap = torch.stack(
            [value.square().sum() for value in projected_scaled]
        ).sum()
        factor = _projection_factor(
            primary_scale,
            auxiliary_scale,
            primary_square,
            projected_square_before_cap,
            cap,
        )
        projected_scaled = tuple(value * factor for value in projected_scaled)
        projected_square = torch.stack([value.square().sum() for value in projected_scaled]).sum()
        projected_dot_scaled = torch.stack(
            [
                (primary_value * projected_value).sum()
                for primary_value, projected_value in zip(primary_scaled, projected_scaled)
            ]
        ).sum()
        for position, projected_value, (_, _, dtype) in zip(
            output_positions, projected_scaled, resolved
        ):
            restored = (projected_value * auxiliary_scale).to(dtype=dtype)
            if not bool(torch.isfinite(restored).all()):
                raise ValueError(f"projected gradient became nonfinite in block {block!r}")
            result_slots[position] = restored

        primary_norm = _scaled_norm(primary_scale, primary_square)
        auxiliary_norm = _scaled_norm(auxiliary_scale, auxiliary_square)
        projected_norm = _scaled_norm(auxiliary_scale, projected_square)
        raw_cosine = None
        if primary_norm > 0.0 and auxiliary_norm > 0.0:
            raw_cosine = float(raw_dot_scaled / torch.sqrt(primary_square * auxiliary_square))
        projected_cosine = None
        if primary_norm > 0.0 and projected_norm > 0.0:
            projected_cosine = float(
                projected_dot_scaled / torch.sqrt(primary_square * projected_square)
            )
        diagnostics[block] = {
            "raw_cosine": raw_cosine,
            "projected_cosine": projected_cosine,
            "primary_norm": primary_norm,
            "auxiliary_norm": auxiliary_norm,
            "projected_norm": projected_norm,
            "raw_dot": _scaled_dot(primary_scale, auxiliary_scale, raw_dot_scaled),
            "projected_dot": _scaled_dot(
                primary_scale, auxiliary_scale, projected_dot_scaled
            ),
            "conflict": conflict,
        }
        projected_by_block[block] = result_slots
    return _ProjectedGradientMap(projected_by_block, diagnostics)


def _support_inputs(
    support_features: Tensor,
    support_labels: Tensor,
    support_tokens: Sequence[str],
    frozen_head_logits: Tensor,
) -> tuple[Tensor, Tensor, tuple[str, ...], Tensor, int]:
    if not isinstance(support_features, Tensor) or support_features.ndim != 2:
        raise ValueError("support features must be a two-dimensional tensor")
    if not support_features.is_floating_point() or not bool(
        torch.isfinite(support_features).all()
    ):
        raise ValueError("support features must be finite and floating point")
    raw_labels = torch.as_tensor(support_labels, device=support_features.device)
    if raw_labels.dtype.is_floating_point or raw_labels.dtype.is_complex or raw_labels.dtype == torch.bool:
        raise ValueError("support labels must use an integer dtype")
    labels = raw_labels.to(dtype=torch.long).view(-1)
    raw_tokens = tuple(support_tokens)
    if any(not isinstance(token, str) or not token for token in raw_tokens):
        raise ValueError("support tokens must be nonempty strings")
    tokens = tuple(raw_tokens)
    logits = torch.as_tensor(
        frozen_head_logits, device=support_features.device
    )
    if labels.numel() != len(support_features) or len(tokens) != len(labels):
        raise ValueError("support features, labels, and tokens must align")
    if not tokens or len(set(tokens)) != len(tokens):
        raise ValueError("support tokens must be nonempty and unique")
    if logits.ndim != 2 or logits.shape[0] != len(labels):
        raise ValueError("frozen head logits must align with support rows")
    if not logits.is_floating_point() or not bool(torch.isfinite(logits).all()):
        raise ValueError("frozen head logits must be finite and floating point")
    classes = torch.unique(labels, sorted=True)
    if not torch.equal(classes, torch.arange(len(classes), device=labels.device)):
        raise ValueError("support labels must be a contiguous zero-based registry")
    if logits.shape[1] != len(classes):
        raise ValueError("frozen head logits must match the registered class count")
    counts = torch.bincount(labels, minlength=len(classes))
    if not bool((counts == counts[0]).all()):
        raise ValueError("support classes must have equal K")
    return support_features, labels, tokens, logits, int(counts[0])


def _cross_fitted_prototype_logits(
    features: Tensor,
    labels: Tensor,
    tokens: Sequence[str],
    *,
    fold_count: int,
    fold_seed: int,
    scale: float,
) -> Tensor:
    folds = stratified_crossfit_indices(
        labels, tokens, fold_count=fold_count, seed=fold_seed
    )
    class_count = int(torch.unique(labels).numel())
    output: Tensor | None = None
    for fold in folds:
        fit = fold.fit_indices.to(device=features.device)
        validation = fold.validation_indices.to(device=features.device)
        fit_labels = labels[fit]
        if any(not bool((fit_labels == class_id).any()) for class_id in range(class_count)):
            raise ValueError("each cross-fit training fold must contain every support class")
        prototypes = torch.stack(
            [features[fit][fit_labels == class_id].mean(dim=0) for class_id in range(class_count)]
        )
        fold_logits = float(scale) * (
            functional.normalize(features[validation], dim=1)
            @ functional.normalize(prototypes, dim=1).T
        )
        if output is None:
            output = torch.empty(
                (len(features), class_count),
                dtype=fold_logits.dtype,
                device=fold_logits.device,
            )
        output[validation] = fold_logits
    if output is None:
        raise ValueError("at least one support cross-fit fold is required")
    return output


def _statistics_loss(
    support_features: Tensor,
    transported_bank: Tensor,
    labels: Tensor,
    *,
    k_shot: int,
    statistic_rank: int,
) -> tuple[Tensor, str]:
    if k_shot == 1:
        mode = "mean_scale"
    elif k_shot < 5:
        mode = "diagonal"
    elif k_shot < 10:
        mode = "low_rank_1"
    else:
        mode = f"low_rank_{min(statistic_rank, k_shot - 1, support_features.shape[1])}"
    losses: list[Tensor] = []

    def safe_square_root(value: Tensor) -> Tensor:
        positive = value > 0.0
        root = torch.where(positive, value, torch.ones_like(value)).sqrt()
        return torch.where(positive, root, torch.zeros_like(root))

    for class_id in torch.unique(labels, sorted=True).tolist():
        support_rows = support_features[labels == int(class_id)].float()
        bank_rows = transported_bank[labels == int(class_id)].float()
        support_mean = support_rows.mean(dim=0)
        bank_mean = bank_rows.mean(dim=0)
        value = functional.mse_loss(support_mean, bank_mean)
        if k_shot == 1:
            support_scale = safe_square_root(support_rows.square().mean())
            bank_scale = safe_square_root(bank_rows.square().mean())
            value = value + (support_scale - bank_scale).square()
        else:
            support_centered = support_rows - support_mean
            bank_centered = bank_rows - bank_mean
            support_variance = support_centered.square().mean(dim=0)
            bank_variance = bank_centered.square().mean(dim=0)
            value = value + functional.mse_loss(support_variance, bank_variance)
            if k_shot >= 5:
                rank = 1 if k_shot < 10 else min(
                    statistic_rank, k_shot - 1, support_features.shape[1]
                )
                normalization = math.sqrt(float(k_shot - 1))
                support_spectrum = torch.linalg.svdvals(support_centered / normalization)[:rank]
                bank_spectrum = torch.linalg.svdvals(bank_centered / normalization)[:rank]
                value = value + functional.mse_loss(support_spectrum, bank_spectrum)
        losses.append(value)
    return torch.stack(losses).mean(), mode


def marc_ot_losses(
    support_features: Tensor,
    support_labels: Tensor,
    support_tokens: Sequence[str],
    frozen_head_logits: Tensor,
    frozen_bank_task_features: Tensor,
    *,
    support_fft_features: Tensor | None = None,
    fold_count: int = 2,
    fold_seed: int = 0,
    ot_epsilon: float = 0.1,
    ot_iterations: int = 80,
    statistic_rank: int = 2,
    prototype_scale: float = 10.0,
    floor_tau: float = 0.1,
    frozen_head_weight: float = 1.0,
    cross_fit_weight: float = 1.0,
    leave_one_out_weight: float = 1.0,
    class_risk_weight: float = 1.0,
    transport_weight: float = 1.0,
    statistics_weight: float = 1.0,
) -> MARCOTDiagnostics:
    """Compose MARC-OT losses from legal support and a frozen task bank only."""

    features, labels, tokens, head_logits, k_shot = _support_inputs(
        support_features, support_labels, support_tokens, frozen_head_logits
    )
    if not isinstance(fold_count, int) or isinstance(fold_count, bool) or fold_count < 2:
        raise ValueError("fold_count must be an integer of at least two")
    if not isinstance(statistic_rank, int) or isinstance(statistic_rank, bool) or statistic_rank < 1:
        raise ValueError("statistic_rank must be a positive integer")
    scale = _finite_positive_scalar(prototype_scale, name="prototype_scale")
    tau = _finite_positive_scalar(floor_tau, name="floor_tau")
    weights = (
        frozen_head_weight,
        cross_fit_weight,
        leave_one_out_weight,
        class_risk_weight,
        transport_weight,
        statistics_weight,
    )
    if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in weights):
        raise ValueError("MARC-OT loss weights must be finite and nonnegative")

    frozen_head_ce = functional.cross_entropy(head_logits, labels)
    if k_shot == 1:
        cross_fit_ce = features.sum() * 0.0
        leave_one_out_ce = features.sum() * 0.0
        class_risk = frozen_class_risk(head_logits, labels)
        cross_fit_mode = "k1_frozen_head"
    else:
        effective_folds = min(fold_count, k_shot)
        if support_fft_features is None:
            cross_fit_logits = _cross_fitted_prototype_logits(
                features,
                labels,
                tokens,
                fold_count=effective_folds,
                fold_seed=int(fold_seed),
                scale=scale,
            )
            cross_fit_ce = functional.cross_entropy(cross_fit_logits, labels)
            class_risk = frozen_class_risk(cross_fit_logits, labels)
            cross_fit_mode = "prototype"
        else:
            fft = torch.as_tensor(
                support_fft_features,
                device=features.device,
                dtype=features.dtype,
            )
            if features.shape[1] != 160 or fft.shape != (len(features), 96):
                raise ValueError("old-only D92 cross-fit requires support [N,160] and FFT [N,96]")
            if int(torch.unique(labels).numel()) != 6:
                raise ValueError("old-only D92 cross-fit requires six registered classes")
            if not bool(torch.isfinite(fft).all()):
                raise ValueError("support FFT features must be finite")
            folds = stratified_crossfit_indices(
                labels,
                tokens,
                fold_count=effective_folds,
                seed=int(fold_seed),
            )
            zeros = features.new_zeros(6)
            d92 = cross_fitted_p3_loss(
                features,
                fft,
                labels,
                folds=folds,
                baseline_class_risk=zeros,
                class_duals=zeros,
                epsilon=zeros,
                rho=0.0,
                beta=0.0,
                tau=tau,
            )
            cross_fit_ce = d92.mean_risk
            class_risk = d92.class_risk
            cross_fit_mode = "d92_old_only"
        leave_one_out_logits = leave_one_out_prototype_logits(
            features, labels, scale=scale
        )
        leave_one_out_ce = functional.cross_entropy(leave_one_out_logits, labels)
    class_risk_loss = tau * torch.logsumexp(class_risk / tau, dim=0)

    transport = support_bank_transport(
        features,
        frozen_bank_task_features,
        epsilon=ot_epsilon,
        iterations=ot_iterations,
    )
    bank = frozen_bank_task_features.detach().to(
        device=features.device, dtype=torch.float32
    )
    cost = torch.cdist(features.float(), bank).square()
    transport_loss = torch.sum(transport * cost)
    row_mass = transport.sum(dim=1, keepdim=True)
    transported_bank = transport @ bank / row_mass
    statistics_loss, statistics_mode = _statistics_loss(
        features,
        transported_bank,
        labels,
        k_shot=k_shot,
        statistic_rank=statistic_rank,
    )
    row_target = transport.new_full((transport.shape[0],), 1.0 / float(transport.shape[0]))
    column_target = transport.new_full(
        (transport.shape[1],), 1.0 / float(transport.shape[1])
    )
    row_error = float((transport.detach().sum(dim=1) - row_target).abs().max())
    column_error = float((transport.detach().sum(dim=0) - column_target).abs().max())
    total = (
        float(frozen_head_weight) * frozen_head_ce
        + float(cross_fit_weight) * cross_fit_ce
        + float(leave_one_out_weight) * leave_one_out_ce
        + float(class_risk_weight) * class_risk_loss
        + float(transport_weight) * transport_loss
        + float(statistics_weight) * statistics_loss
    )
    components = (
        total,
        frozen_head_ce,
        cross_fit_ce,
        leave_one_out_ce,
        class_risk,
        class_risk_loss,
        transport_loss,
        statistics_loss,
    )
    if any(not bool(torch.isfinite(component).all()) for component in components):
        raise ValueError("MARC-OT loss became nonfinite")
    return MARCOTDiagnostics(
        total=total,
        frozen_head_ce=frozen_head_ce,
        cross_fit_ce=cross_fit_ce,
        leave_one_out_ce=leave_one_out_ce,
        class_risk=class_risk,
        class_risk_loss=class_risk_loss,
        transport_loss=transport_loss,
        statistics_loss=statistics_loss,
        k_shot=k_shot,
        statistics_mode=statistics_mode,
        cross_fit_mode=cross_fit_mode,
        transport_row_error=row_error,
        transport_column_error=column_error,
    )


__all__ = [
    "MARCOTDiagnostics",
    "blockwise_primary_projection",
    "marc_ot_losses",
    "support_bank_transport",
]
