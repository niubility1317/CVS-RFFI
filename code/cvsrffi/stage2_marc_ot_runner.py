"""Support-only MARC-OT Stage2-B runner and frozen per-query inference."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
import inspect
import math
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as functional

from .meta_weight_bank import parameter_block_key
from .stage2_marc_ot import blockwise_primary_projection, marc_ot_losses
from .stage2_wiser_p3 import stratified_crossfit_indices


MARCOT_PROGRESSIVE_STAGES = (
    "norm_fusion_projection",
    "t3_f3_identity",
    "t2_f2",
    "t1_f1",
)
MARCOT_FORMAL_ARMS = ("R0", "R1", "R2", "R4", "R6", "R8")


@dataclass(frozen=True)
class MARCOTRunnerConfig:
    """Frozen support-only optimization and rollback policy."""

    fold_count: int = 5
    stage_steps: tuple[int, int, int, int] = (1500, 2000, 2500, 3000)
    learning_rate_min: float = 1.0e-5
    learning_rate_max: float = 3.0e-4
    ot_epsilon: float = 0.1
    ot_iterations: int = 80
    ratio_cap: float = 0.5
    interpolation_grid: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25, 0.0)
    support_ba_tolerance: float = 0.0
    support_floor_tolerance: float = 0.0
    trust_weight: float = 1.0e-4
    seed: int = 713102

    def __post_init__(self) -> None:
        if not isinstance(self.fold_count, int) or isinstance(self.fold_count, bool) or self.fold_count < 2:
            raise ValueError("fold_count must be an integer of at least two")
        if len(self.stage_steps) != 4 or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in self.stage_steps
        ):
            raise ValueError("stage_steps must contain four nonnegative integers")
        if self.learning_rate_min <= 0.0 or self.learning_rate_min >= self.learning_rate_max:
            raise ValueError("learning-rate bounds must be positive and ordered")
        finite_positive = (self.ot_epsilon, self.learning_rate_max)
        if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in finite_positive):
            raise ValueError("positive runner values must be finite")
        if not isinstance(self.ot_iterations, int) or isinstance(self.ot_iterations, bool) or self.ot_iterations < 1:
            raise ValueError("ot_iterations must be a positive integer")
        if not math.isfinite(float(self.ratio_cap)) or float(self.ratio_cap) < 0.0:
            raise ValueError("ratio_cap must be finite and nonnegative")
        if not self.interpolation_grid or any(
            not math.isfinite(float(alpha)) or not 0.0 <= float(alpha) <= 1.0
            for alpha in self.interpolation_grid
        ):
            raise ValueError("interpolation_grid must contain finite [0,1] values")
        if 0.0 not in tuple(float(alpha) for alpha in self.interpolation_grid):
            raise ValueError("interpolation_grid must contain alpha=0")
        for value in (
            self.support_ba_tolerance,
            self.support_floor_tolerance,
            self.trust_weight,
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError("support tolerances and trust weight must be finite and nonnegative")


@dataclass(frozen=True)
class SupportSafeSelection:
    selected_alpha: float
    state: Mapping[str, Tensor]
    duals: Mapping[str, Tensor]
    support_metrics: Mapping[str, Any]
    query_rows_used: int = 0


@dataclass(frozen=True)
class MARCOTTrainingAudit:
    arm: str
    selected_alpha: float
    initial_selected_alpha: float
    stage_selected_alphas: tuple[float, ...]
    optimizer_steps: int
    query_rows_used: int
    stage_audits: tuple[Mapping[str, Any], ...]
    final_duals: Mapping[str, tuple[float | int, ...]]
    config: Mapping[str, Any]
    training_seconds: float
    peak_cuda_bytes: int | None
    reached_parameter_names: tuple[str, ...] = field(default_factory=tuple)


def _clone_tensors(values: Mapping[str, Tensor], *, context: str) -> dict[str, Tensor]:
    result: dict[str, Tensor] = {}
    for name, value in values.items():
        if not isinstance(name, str) or not name or not isinstance(value, Tensor):
            raise ValueError(f"{context} must map nonempty names to tensors")
        result[name] = value.detach().clone()
    return result


def _validate_matching_state(
    base: Mapping[str, Tensor], candidate: Mapping[str, Tensor], *, context: str
) -> None:
    if set(base) != set(candidate):
        raise ValueError(f"{context} members differ")
    for name, base_value in base.items():
        candidate_value = candidate[name]
        if base_value.shape != candidate_value.shape or base_value.dtype != candidate_value.dtype:
            raise ValueError(f"{context} geometry differs: {name}")
        if candidate_value.is_floating_point() and not bool(torch.isfinite(candidate_value).all()):
            raise ValueError(f"{context} contains nonfinite values: {name}")


def select_support_safe_state(
    base_state: Mapping[str, Tensor],
    candidate_state: Mapping[str, Tensor],
    *,
    base_duals: Mapping[str, Tensor],
    candidate_duals: Mapping[str, Tensor],
    evaluator: Callable[[Mapping[str, Tensor], Mapping[str, Tensor]], Mapping[str, Any] | bool],
    grid: Sequence[float],
    trainable_parameter_names: Sequence[str],
) -> SupportSafeSelection:
    """Select only with support evidence; alpha=0 is an exact tensor fallback."""

    base = _clone_tensors(base_state, context="base state")
    candidate = _clone_tensors(candidate_state, context="candidate state")
    dual_base = _clone_tensors(base_duals, context="base duals")
    dual_candidate = _clone_tensors(candidate_duals, context="candidate duals")
    _validate_matching_state(base, candidate, context="interpolation state")
    _validate_matching_state(dual_base, dual_candidate, context="interpolation dual")
    alphas = tuple(float(value) for value in grid)
    if not alphas or any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in alphas):
        raise ValueError("interpolation grid must contain finite [0,1] values")
    if 0.0 not in alphas:
        raise ValueError("interpolation grid must contain alpha=0")
    trainable = set(str(name) for name in trainable_parameter_names)
    if not trainable.issubset(base):
        raise ValueError("interpolation trainable registry is outside base state")

    fallback_metrics: Mapping[str, Any] = {"safe": False}
    for alpha in alphas:
        state: dict[str, Tensor] = {}
        for name, base_value in base.items():
            candidate_value = candidate[name]
            if alpha != 0.0 and name in trainable and base_value.is_floating_point():
                state[name] = base_value + alpha * (candidate_value - base_value)
            else:
                state[name] = base_value.detach().clone()
        duals: dict[str, Tensor] = {}
        for name, base_value in dual_base.items():
            candidate_value = dual_candidate[name]
            if alpha != 0.0 and base_value.is_floating_point():
                duals[name] = base_value + alpha * (candidate_value - base_value)
            else:
                duals[name] = base_value.detach().clone()
        observed = evaluator(state, duals)
        metrics: Mapping[str, Any] = (
            {"safe": bool(observed)} if isinstance(observed, bool) else dict(observed)
        )
        if alpha == 0.0:
            fallback_metrics = metrics
        if alpha != 0.0 and bool(metrics.get("safe", False)):
            return SupportSafeSelection(alpha, state, duals, metrics)
    return SupportSafeSelection(0.0, base, dual_base, fallback_metrics)


def _forward_identity(model: nn.Module, values: Tensor) -> tuple[Tensor, Tensor]:
    try:
        parameters = inspect.signature(model.forward).parameters
    except (TypeError, ValueError) as error:
        raise ValueError("cannot inspect MARC-OT model.forward") from error
    kwargs: dict[str, Any] = {}
    if "return_aux" in parameters:
        kwargs["return_aux"] = True
    for label_name in ("y_tx", "y"):
        if label_name in parameters:
            kwargs[label_name] = None
            break
    output = model(values, **kwargs)
    if not isinstance(output, Mapping):
        raise ValueError("MARC-OT model must return an auxiliary mapping")
    logits = output.get("tx_logits", output.get("logits"))
    features = output.get("z_id")
    if (
        not isinstance(logits, Tensor)
        or not isinstance(features, Tensor)
        or logits.ndim != 2
        or features.ndim != 2
        or logits.shape[0] != values.shape[0]
        or features.shape[0] != values.shape[0]
        or not bool(torch.isfinite(logits).all())
        or not bool(torch.isfinite(features).all())
    ):
        raise ValueError("MARC-OT identity output geometry or finiteness drift")
    return logits, features


def _validate_support(
    support_iq: Tensor, support_labels: Tensor, support_tokens: Sequence[str]
) -> tuple[Tensor, Tensor, tuple[str, ...]]:
    values = torch.as_tensor(support_iq)
    raw_labels = torch.as_tensor(support_labels, device=values.device)
    if raw_labels.dtype.is_floating_point or raw_labels.dtype.is_complex or raw_labels.dtype == torch.bool:
        raise ValueError("support labels must use an integer dtype")
    labels = raw_labels.to(dtype=torch.long).view(-1)
    tokens = tuple(support_tokens)
    if values.ndim < 2 or values.shape[0] != labels.numel() or not bool(torch.isfinite(values).all()):
        raise ValueError("support values and labels are invalid")
    if len(tokens) != len(labels) or len(set(tokens)) != len(tokens):
        raise ValueError("support tokens must be unique and align with support rows")
    if any(not isinstance(token, str) or not token for token in tokens):
        raise ValueError("support tokens must be nonempty strings")
    classes = torch.unique(labels, sorted=True)
    if not torch.equal(classes, torch.arange(len(classes), device=labels.device)):
        raise ValueError("support labels must form a contiguous zero-based registry")
    counts = torch.bincount(labels, minlength=len(classes))
    if not bool((counts == counts[0]).all()):
        raise ValueError("support classes must have equal K")
    return values, labels, tokens


def _stage_reached_names(model: nn.Module, stage_index: int) -> tuple[str, ...]:
    allowed_blocks_by_stage = (
        {"fusion", "time_projection", "frequency_projection"},
        {"t3", "f3", "identity_mapping"},
        {"t2", "f2"},
        {"t1", "f1"},
    )
    reached = set().union(*allowed_blocks_by_stage[: stage_index + 1])
    names: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.is_floating_point():
            continue
        block = parameter_block_key(name)
        is_identity_norm = name.startswith("id_backbone.") and "norm" in name.lower().split(".")
        if block in reached or (stage_index >= 0 and is_identity_norm):
            names.append(name)
    if not names:
        raise ValueError(f"MARC-OT stage has no trainable parameters: {MARCOT_PROGRESSIVE_STAGES[stage_index]}")
    return tuple(names)


def _load_state(model: nn.Module, state: Mapping[str, Tensor]) -> None:
    model.load_state_dict(dict(state), strict=True)


def _support_crossfit_metrics(
    model: nn.Module,
    values: Tensor,
    labels: Tensor,
    tokens: tuple[str, ...],
    *,
    fold_count: int,
    seed: int,
) -> dict[str, float | bool]:
    model.eval()
    folds = stratified_crossfit_indices(
        labels,
        tokens,
        fold_count=min(int(fold_count), int(torch.bincount(labels).min())),
        seed=int(seed),
    )
    with torch.inference_mode():
        _, features = _forward_identity(model, values)
        predictions = torch.empty_like(labels)
        for fold in folds:
            fit = fold.fit_indices.to(labels.device)
            validation = fold.validation_indices.to(labels.device)
            fit_labels = labels[fit]
            prototypes = torch.stack(
                [features[fit][fit_labels == class_id].mean(dim=0) for class_id in range(len(torch.unique(labels)))]
            )
            logits = functional.normalize(features[validation].float(), dim=1) @ functional.normalize(
                prototypes.float(), dim=1
            ).T
            predictions[validation] = logits.argmax(dim=1)
    per_class = torch.stack(
        [(predictions[labels == class_id] == class_id).float().mean() for class_id in range(len(torch.unique(labels)))]
    )
    return {
        "safe": bool(torch.isfinite(per_class).all()),
        "oof_ba": float(per_class.mean()),
        "oof_floor": float(per_class.min()),
    }


def _default_fold_metrics(
    model: nn.Module,
    state: Mapping[str, Tensor],
    fit_iq: Tensor,
    fit_labels: Tensor,
    validation_iq: Tensor,
    validation_labels: Tensor,
) -> Mapping[str, Any]:
    _load_state(model, state)
    model.eval()
    with torch.inference_mode():
        _, fit_features = _forward_identity(model, fit_iq)
        _, validation_features = _forward_identity(model, validation_iq)
        class_count = int(torch.unique(fit_labels).numel())
        prototypes = torch.stack(
            [
                fit_features[fit_labels == class_id].mean(dim=0)
                for class_id in range(class_count)
            ]
        )
        logits = functional.normalize(validation_features.float(), dim=1) @ functional.normalize(
            prototypes.float(), dim=1
        ).T
        predictions = logits.argmax(dim=1)
    correct = tuple(
        int((predictions[validation_labels == class_id] == class_id).sum())
        for class_id in range(class_count)
    )
    rows = tuple(
        int((validation_labels == class_id).sum()) for class_id in range(class_count)
    )
    return {
        "safe": bool(torch.isfinite(logits).all()) and all(value > 0 for value in rows),
        "per_class_correct": correct,
        "per_class_rows": rows,
    }


def _aggregate_fold_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("MARC-OT cross-fit needs at least one fold metric")
    safe = all(bool(row.get("safe", False)) for row in rows)
    if all("per_class_correct" in row and "per_class_rows" in row for row in rows):
        class_count = len(tuple(rows[0]["per_class_rows"]))
        correct = np.zeros(class_count, dtype=np.float64)
        counts = np.zeros(class_count, dtype=np.float64)
        for row in rows:
            current_correct = np.asarray(row["per_class_correct"], dtype=np.float64)
            current_counts = np.asarray(row["per_class_rows"], dtype=np.float64)
            if current_correct.shape != (class_count,) or current_counts.shape != (class_count,):
                raise ValueError("MARC-OT cross-fit class metric geometry drift")
            correct += current_correct
            counts += current_counts
        if bool((counts <= 0.0).any()):
            raise ValueError("MARC-OT cross-fit lacks registered-class validation rows")
        per_class = correct / counts
        return {
            "safe": safe and bool(np.isfinite(per_class).all()),
            "oof_ba": float(per_class.mean()),
            "oof_floor": float(per_class.min()),
            "per_class_accuracy": per_class.tolist(),
        }
    ba = [float(row["oof_ba"]) for row in rows if "oof_ba" in row]
    floor = [float(row["oof_floor"]) for row in rows if "oof_floor" in row]
    return {
        "safe": safe,
        "oof_ba": float(np.mean(ba)) if ba else float("nan"),
        "oof_floor": float(min(floor)) if floor else float("nan"),
    }


def _interpolate_for_alpha(
    base_state: Mapping[str, Tensor],
    candidate_state: Mapping[str, Tensor],
    base_duals: Mapping[str, Tensor],
    candidate_duals: Mapping[str, Tensor],
    *,
    alpha: float,
    trainable_parameter_names: Sequence[str],
) -> tuple[Mapping[str, Tensor], Mapping[str, Tensor]]:
    selection = select_support_safe_state(
        base_state,
        candidate_state,
        base_duals=base_duals,
        candidate_duals=candidate_duals,
        evaluator=lambda _state, _duals: True,
        grid=((float(alpha), 0.0) if float(alpha) != 0.0 else (0.0,)),
        trainable_parameter_names=trainable_parameter_names,
    )
    return selection.state, selection.duals


def _default_stage_update(
    model: nn.Module,
    stage: str,
    trainable_names: Sequence[str],
    steps: int,
    duals: Mapping[str, Tensor],
    *,
    values: Tensor,
    labels: Tensor,
    tokens: tuple[str, ...],
    arm: str,
    config: MARCOTRunnerConfig,
    bank_task_features: Tensor | None,
    calibration_feature_transform: Callable[
        [nn.Module, Tensor, Tensor, tuple[str, ...]], Tensor
    ]
    | None,
    block_learning_rates: Mapping[str, float] | None,
    original_base: Mapping[str, Tensor],
) -> tuple[Mapping[str, Tensor], Mapping[str, Tensor]]:
    del stage
    named = dict(model.named_parameters())
    selected = [named[name] for name in trainable_names]
    for name, parameter in named.items():
        parameter.requires_grad_(name in set(trainable_names))
    if steps == 0:
        return _clone_tensors(model.state_dict(), context="zero-step state"), _clone_tensors(duals, context="zero-step duals")
    resolved_learning_rates = resolve_block_learning_rates(
        trainable_names, block_learning_rates, config=config
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": [parameter], "lr": learning_rate}
            for parameter, learning_rate in zip(selected, resolved_learning_rates, strict=True)
        ],
        weight_decay=0.0,
    )
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        logits, features = _forward_identity(model, values)
        if logits.shape[1] != int(torch.unique(labels).numel()):
            raise ValueError("frozen head registry differs from support registry")
        if arm == "R1":
            loss = functional.cross_entropy(logits, labels)
            loss.backward()
        else:
            calibration_features = (
                features
                if calibration_feature_transform is None
                else calibration_feature_transform(model, values, labels, tokens)
            )
            if (
                not isinstance(calibration_features, Tensor)
                or calibration_features.ndim != 2
                or calibration_features.shape[0] != len(labels)
                or not bool(torch.isfinite(calibration_features).all())
            ):
                raise ValueError("support calibration feature transform is invalid")
            bank = (
                calibration_features.detach()
                if bank_task_features is None
                else torch.as_tensor(
                    bank_task_features,
                    device=calibration_features.device,
                    dtype=calibration_features.dtype,
                ).detach()
            )
            if arm in {"R6", "R8"} and bank_task_features is None:
                raise ValueError(f"{arm} requires frozen bank task features")
            diagnostics = marc_ot_losses(
                calibration_features,
                labels,
                tokens,
                logits,
                bank,
                fold_count=int(config.fold_count),
                fold_seed=int(config.seed),
                ot_epsilon=float(config.ot_epsilon),
                ot_iterations=int(config.ot_iterations),
                transport_weight=1.0 if arm in {"R6", "R8"} else 0.0,
                statistics_weight=1.0 if arm in {"R6", "R8"} else 0.0,
            )
            trust = sum(
                (named[name] - original_base[name].to(named[name].device)).float().square().mean()
                for name in trainable_names
            )
            primary = (
                diagnostics.frozen_head_ce
                + diagnostics.cross_fit_ce
                + diagnostics.leave_one_out_ce
                + diagnostics.class_risk_loss
                + float(config.trust_weight) * trust
            )
            auxiliary = diagnostics.transport_loss + diagnostics.statistics_loss
            if arm != "R8":
                (primary + (auxiliary if arm == "R6" else 0.0)).backward()
            else:
                primary_gradients = torch.autograd.grad(
                    primary, selected, retain_graph=True, allow_unused=True
                )
                auxiliary_gradients = torch.autograd.grad(
                    auxiliary, selected, allow_unused=True
                )
                combined = combine_blockwise_gradients(
                    trainable_names,
                    primary_gradients,
                    auxiliary_gradients,
                    ratio_cap=float(config.ratio_cap),
                )
                for index, gradient in enumerate(combined):
                    selected[index].grad = None if gradient is None else gradient.detach().clone()
        for name, parameter in zip(trainable_names, selected, strict=True):
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                raise RuntimeError(f"MARC-OT gradient became nonfinite: {name}")
        optimizer.step()
        if any(not bool(torch.isfinite(parameter).all()) for parameter in selected):
            raise RuntimeError("MARC-OT parameter became nonfinite")
    return _clone_tensors(model.state_dict(), context="candidate state"), _clone_tensors(duals, context="candidate duals")


def resolve_block_learning_rates(
    parameter_names: Sequence[str],
    block_learning_rates: Mapping[str, float] | None,
    *,
    config: MARCOTRunnerConfig,
) -> tuple[float, ...]:
    """Resolve frozen task-conditioned block rates within preregistered bounds."""

    if not isinstance(config, MARCOTRunnerConfig):
        raise ValueError("config must be MARCOTRunnerConfig")
    mapping = {} if block_learning_rates is None else dict(block_learning_rates)
    for block, value in mapping.items():
        if not isinstance(block, str) or not block:
            raise ValueError("block learning-rate names must be nonempty strings")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError("block learning rates must be numeric")
        numeric = float(value)
        if (
            not math.isfinite(numeric)
            or numeric < float(config.learning_rate_min)
            or numeric > float(config.learning_rate_max)
        ):
            raise ValueError("block learning rates are outside frozen bounds")
    result = []
    for name in parameter_names:
        block = parameter_block_key(name)
        result.append(float(mapping.get(block, config.learning_rate_max)))
    return tuple(result)


def combine_blockwise_gradients(
    parameter_names: Sequence[str],
    primary: Sequence[Tensor | None],
    calibration: Sequence[Tensor | None],
    *,
    ratio_cap: float,
) -> tuple[Tensor | None, ...]:
    """Return `g_task + projected(g_cal)` without cross-block interference."""

    names = tuple(parameter_names)
    primary_values = tuple(primary)
    calibration_values = tuple(calibration)
    if not names or len(names) != len(primary_values) or len(names) != len(calibration_values):
        raise ValueError("gradient names and values must be nonempty and aligned")
    canonical_indices = [index for index, name in enumerate(names) if parameter_block_key(name) is not None]
    primary_map = {names[index]: [primary_values[index]] for index in canonical_indices}
    calibration_map = {names[index]: [calibration_values[index]] for index in canonical_indices}
    projected = (
        blockwise_primary_projection(primary_map, calibration_map, ratio_cap=ratio_cap)
        if primary_map
        else {}
    )
    block_offsets: dict[str, int] = {}
    result: list[Tensor | None] = []
    for index, name in enumerate(names):
        task_gradient = primary_values[index]
        block = parameter_block_key(name)
        if block is None:
            calibration_gradient = calibration_values[index]
        else:
            offset = block_offsets.get(block, 0)
            calibration_gradient = projected[block][offset]
            block_offsets[block] = offset + 1
        if task_gradient is None and calibration_gradient is None:
            combined = None
        elif task_gradient is None:
            combined = calibration_gradient
        elif calibration_gradient is None:
            combined = task_gradient
        else:
            combined = task_gradient + calibration_gradient
        if combined is not None and not bool(torch.isfinite(combined).all()):
            raise RuntimeError(f"combined MARC-OT gradient became nonfinite: {name}")
        result.append(combined)
    return tuple(result)


def _serialize_duals(duals: Mapping[str, Tensor]) -> dict[str, tuple[float | int, ...]]:
    result: dict[str, tuple[float | int, ...]] = {}
    for name, value in duals.items():
        flat = value.detach().cpu().view(-1)
        result[name] = tuple(
            float(item) if value.is_floating_point() else int(item) for item in flat
        )
    return result


def train_marc_ot_arm(
    model: nn.Module,
    support_iq: Tensor,
    support_labels: Tensor,
    support_tokens: Sequence[str],
    *,
    arm: str,
    config: MARCOTRunnerConfig,
    bank_task_features: Tensor | None = None,
    calibration_feature_transform: Callable[
        [nn.Module, Tensor, Tensor, tuple[str, ...]], Tensor
    ]
    | None = None,
    block_learning_rates: Mapping[str, float] | None = None,
    block_learning_rate_factory: Callable[
        [Tensor, Tensor, tuple[str, ...], str], Mapping[str, float]
    ]
    | None = None,
    initial_state_factory: Callable[
        [Tensor, Tensor, tuple[str, ...], str], Mapping[str, Tensor]
    ]
    | None = None,
    initial_duals: Mapping[str, Tensor] | None = None,
    stage_update: Callable[..., tuple[Mapping[str, Tensor], Mapping[str, Tensor]]]
    | None = None,
    support_evaluator: Callable[..., Mapping[str, Any] | bool] | None = None,
) -> MARCOTTrainingAudit:
    """Adapt from legal support only, select by support cross-fit, then refreeze."""

    arm_value = str(arm).upper()
    if arm_value not in MARCOT_FORMAL_ARMS:
        raise ValueError("MARC-OT arm is outside the frozen R matrix")
    if not isinstance(config, MARCOTRunnerConfig):
        raise ValueError("config must be MARCOTRunnerConfig")
    if block_learning_rates is not None and block_learning_rate_factory is not None:
        raise ValueError("static and fit-scope block learning rates are mutually exclusive")
    values, labels, tokens = _validate_support(support_iq, support_labels, support_tokens)
    original_base = _clone_tensors(model.state_dict(), context="base model state")
    original_duals = (
        {"class_duals": torch.zeros(int(torch.unique(labels).numel()), device=values.device)}
        if initial_duals is None
        else _clone_tensors(initial_duals, context="initial duals")
    )
    started = time.perf_counter()
    stage_audits: list[Mapping[str, Any]] = []
    reached: tuple[str, ...] = ()
    selected_alpha = 0.0
    initial_selected_alpha = 0.0
    stage_selected_alphas: list[float] = []
    optimizer_steps = 0
    current_duals = _clone_tensors(original_duals, context="current duals")

    def refreeze() -> None:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    try:
        _load_state(model, original_base)
        refreeze()
        if arm_value == "R0":
            _load_state(model, original_base)
            return MARCOTTrainingAudit(
                arm=arm_value,
                selected_alpha=0.0,
                initial_selected_alpha=0.0,
                stage_selected_alphas=(),
                optimizer_steps=0,
                query_rows_used=0,
                stage_audits=(),
                final_duals=_serialize_duals(original_duals),
                config=asdict(config),
                training_seconds=float(time.perf_counter() - started),
                peak_cuda_bytes=(int(torch.cuda.max_memory_allocated(values.device)) if values.is_cuda else None),
            )

        class_minimum = int(torch.bincount(labels).min())
        if class_minimum >= 2:
            raw_folds = stratified_crossfit_indices(
                labels,
                tokens,
                fold_count=min(int(config.fold_count), class_minimum),
                seed=int(config.seed),
            )
            fold_indices = tuple(
                (
                    fold.fit_indices.to(device=labels.device, dtype=torch.long),
                    fold.validation_indices.to(device=labels.device, dtype=torch.long),
                )
                for fold in raw_folds
            )
        elif support_evaluator is not None and stage_update is not None:
            # A K=1 unit cannot form a fitted class prototype.  Keep the legacy
            # callback-only seam usable without ever putting validation rows in
            # its selection update; formal pilot units use K=10.
            fold_indices = (
                (
                    torch.empty(0, dtype=torch.long, device=labels.device),
                    torch.arange(len(labels), dtype=torch.long, device=labels.device),
                ),
            )
        else:
            raise ValueError("MARC-OT adapter cross-fit requires at least two support rows per class")

        def fold_payload(indices: Tensor) -> tuple[Tensor, Tensor, tuple[str, ...]]:
            selected = tuple(tokens[int(index)] for index in indices.detach().cpu().tolist())
            return values[indices], labels[indices], selected

        def evaluate_fold(
            state: Mapping[str, Tensor],
            duals: Mapping[str, Tensor],
            fit_indices: Tensor,
            validation_indices: Tensor,
        ) -> Mapping[str, Any]:
            fit_iq, fit_labels, fit_tokens = fold_payload(fit_indices)
            validation_iq, validation_labels, validation_tokens = fold_payload(
                validation_indices
            )
            if support_evaluator is None:
                return _default_fold_metrics(
                    model,
                    state,
                    fit_iq,
                    fit_labels,
                    validation_iq,
                    validation_labels,
                )
            observed = support_evaluator(
                state,
                duals,
                fit_iq,
                fit_labels,
                fit_tokens,
                validation_iq,
                validation_labels,
                validation_tokens,
            )
            return {"safe": bool(observed)} if isinstance(observed, bool) else dict(observed)

        def aggregate_states(
            states: Sequence[Mapping[str, Tensor]],
            duals: Sequence[Mapping[str, Tensor]],
        ) -> dict[str, Any]:
            if len(states) != len(fold_indices) or len(duals) != len(fold_indices):
                raise ValueError("MARC-OT cross-fit state count drift")
            return _aggregate_fold_metrics(
                [
                    evaluate_fold(state, dual, fit, validation)
                    for state, dual, (fit, validation) in zip(
                        states, duals, fold_indices, strict=True
                    )
                ]
            )

        def select_crossfit_candidate(
            stage_base_state: Mapping[str, Tensor],
            stage_base_duals: Mapping[str, Tensor],
            candidate_states: Sequence[Mapping[str, Tensor]],
            candidate_duals: Sequence[Mapping[str, Tensor]],
            *,
            trainable_parameter_names: Sequence[str],
        ) -> tuple[float, Mapping[str, Any]]:
            base_states = [stage_base_state for _ in fold_indices]
            base_dual_rows = [stage_base_duals for _ in fold_indices]
            baseline_metrics = aggregate_states(base_states, base_dual_rows)
            fallback_metrics: Mapping[str, Any] = baseline_metrics
            for alpha in tuple(float(value) for value in config.interpolation_grid):
                interpolated = [
                    _interpolate_for_alpha(
                        stage_base_state,
                        candidate_state,
                        stage_base_duals,
                        dual_candidate,
                        alpha=alpha,
                        trainable_parameter_names=trainable_parameter_names,
                    )
                    for candidate_state, dual_candidate in zip(
                        candidate_states, candidate_duals, strict=True
                    )
                ]
                metrics = aggregate_states(
                    [row[0] for row in interpolated],
                    [row[1] for row in interpolated],
                )
                if support_evaluator is None:
                    metrics["safe"] = bool(metrics.get("safe", False)) and (
                        float(metrics["oof_ba"])
                        >= float(baseline_metrics["oof_ba"])
                        - float(config.support_ba_tolerance)
                        and float(metrics["oof_floor"])
                        >= float(baseline_metrics["oof_floor"])
                        - float(config.support_floor_tolerance)
                    )
                if alpha == 0.0:
                    fallback_metrics = metrics
                if alpha != 0.0 and bool(metrics.get("safe", False)):
                    return alpha, metrics
            return 0.0, fallback_metrics

        def update_from_stage_base(
            stage_base_state: Mapping[str, Tensor],
            stage_base_duals: Mapping[str, Tensor],
            stage: str,
            trainable_names: Sequence[str],
            fit_indices: Tensor,
            *,
            fit_scope: str,
        ) -> tuple[Mapping[str, Tensor], Mapping[str, Tensor]]:
            _load_state(model, stage_base_state)
            model.train()
            trainable = set(trainable_names)
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(name in trainable)
            fit_iq, fit_labels, fit_tokens = fold_payload(fit_indices)
            scoped_learning_rates = block_learning_rates
            if block_learning_rate_factory is not None:
                observed_learning_rates = block_learning_rate_factory(
                    fit_iq, fit_labels, fit_tokens, fit_scope
                )
                if not isinstance(observed_learning_rates, Mapping):
                    raise ValueError("fit-scope block learning-rate factory is invalid")
                scoped_learning_rates = dict(observed_learning_rates)
                resolve_block_learning_rates(
                    trainable_names, scoped_learning_rates, config=config
                )
            if stage_update is not None:
                state, duals = stage_update(
                    model,
                    stage,
                    trainable_names,
                    int(config.stage_steps[MARCOT_PROGRESSIVE_STAGES.index(stage)]),
                    _clone_tensors(stage_base_duals, context="stage update duals"),
                    fit_iq,
                    fit_labels,
                    fit_tokens,
                    fit_scope,
                )
            else:
                state, duals = _default_stage_update(
                    model,
                    stage,
                    trainable_names,
                    int(config.stage_steps[MARCOT_PROGRESSIVE_STAGES.index(stage)]),
                    stage_base_duals,
                    values=fit_iq,
                    labels=fit_labels,
                    tokens=fit_tokens,
                    arm=arm_value,
                    config=config,
                    bank_task_features=bank_task_features,
                    calibration_feature_transform=calibration_feature_transform,
                    block_learning_rates=scoped_learning_rates,
                    original_base=original_base,
                )
            cloned_state = _clone_tensors(state, context="stage candidate state")
            cloned_duals = _clone_tensors(duals, context="stage candidate duals")
            _validate_matching_state(
                stage_base_state, cloned_state, context="stage candidate state"
            )
            _validate_matching_state(
                stage_base_duals, cloned_duals, context="stage candidate duals"
            )
            return cloned_state, cloned_duals

        current_state = _clone_tensors(original_base, context="current state")
        if initial_state_factory is not None and arm_value in {"R4", "R6", "R8"}:
            fold_bank_states: list[Mapping[str, Tensor]] = []
            for fit, _validation in fold_indices:
                _load_state(model, original_base)
                fit_iq, fit_labels, fit_tokens = fold_payload(fit)
                bank_state = _clone_tensors(
                    initial_state_factory(
                        fit_iq, fit_labels, fit_tokens, "crossfit"
                    ),
                    context="fold initial bank state",
                )
                _validate_matching_state(
                    original_base, bank_state, context="fold initial bank state"
                )
                fold_bank_states.append(bank_state)
            initial_selected_alpha, initial_metrics = select_crossfit_candidate(
                current_state,
                current_duals,
                fold_bank_states,
                [current_duals for _ in fold_indices],
                trainable_parameter_names=tuple(
                    name for name in current_state if parameter_block_key(name) is not None
                ),
            )
            del initial_metrics
            if initial_selected_alpha != 0.0:
                _load_state(model, original_base)
                bank_state = _clone_tensors(
                    initial_state_factory(values, labels, tokens, "full_support"),
                    context="full-support initial bank state",
                )
                _validate_matching_state(
                    original_base, bank_state, context="full-support initial bank state"
                )
                current_state, current_duals = _interpolate_for_alpha(
                    current_state,
                    bank_state,
                    current_duals,
                    current_duals,
                    alpha=initial_selected_alpha,
                    trainable_parameter_names=tuple(
                        name for name in current_state if parameter_block_key(name) is not None
                    ),
                )
                current_state = dict(current_state)
                current_duals = dict(current_duals)
                selected_alpha = initial_selected_alpha
            _load_state(model, current_state)

        for index, stage in enumerate(MARCOT_PROGRESSIVE_STAGES):
            reached = _stage_reached_names(model, index)
            stage_base_state = _clone_tensors(current_state, context="stage base state")
            stage_base_duals = _clone_tensors(current_duals, context="stage base duals")
            fold_candidates = [
                update_from_stage_base(
                    stage_base_state,
                    stage_base_duals,
                    stage,
                    reached,
                    fit,
                    fit_scope="crossfit",
                )
                for fit, _validation in fold_indices
            ]
            optimizer_steps += int(config.stage_steps[index]) * len(fold_indices)
            stage_alpha, support_metrics = select_crossfit_candidate(
                stage_base_state,
                stage_base_duals,
                [row[0] for row in fold_candidates],
                [row[1] for row in fold_candidates],
                trainable_parameter_names=reached,
            )
            stage_selected_alphas.append(stage_alpha)
            if stage_alpha != 0.0:
                full_candidate_state, full_candidate_duals = update_from_stage_base(
                    stage_base_state,
                    stage_base_duals,
                    stage,
                    reached,
                    torch.arange(len(labels), dtype=torch.long, device=labels.device),
                    fit_scope="full_support",
                )
                current_state, current_duals = _interpolate_for_alpha(
                    stage_base_state,
                    full_candidate_state,
                    stage_base_duals,
                    full_candidate_duals,
                    alpha=stage_alpha,
                    trainable_parameter_names=reached,
                )
                current_state = dict(current_state)
                current_duals = dict(current_duals)
                selected_alpha = stage_alpha
                optimizer_steps += int(config.stage_steps[index])
            else:
                current_state = stage_base_state
                current_duals = stage_base_duals
            stage_audits.append(
                {
                    "stage": stage,
                    "selected_alpha": stage_alpha,
                    "trainable_parameter_names": list(reached),
                    "support_metrics": dict(support_metrics),
                    "crossfit_fold_count": len(fold_indices),
                    "optimizer_steps": int(config.stage_steps[index])
                    * (len(fold_indices) + (1 if stage_alpha != 0.0 else 0)),
                    "query_rows_used": 0,
                }
            )
            _load_state(model, current_state)
        if selected_alpha == 0.0:
            current_state = _clone_tensors(original_base, context="original fallback state")
            current_duals = _clone_tensors(original_duals, context="original fallback duals")
            _load_state(model, current_state)
        return MARCOTTrainingAudit(
            arm=arm_value,
            selected_alpha=selected_alpha,
            initial_selected_alpha=initial_selected_alpha,
            stage_selected_alphas=tuple(stage_selected_alphas),
            optimizer_steps=optimizer_steps,
            query_rows_used=0,
            stage_audits=tuple(stage_audits),
            final_duals=_serialize_duals(current_duals),
            config=asdict(config),
            training_seconds=float(time.perf_counter() - started),
            peak_cuda_bytes=(int(torch.cuda.max_memory_allocated(values.device)) if values.is_cuda else None),
            reached_parameter_names=reached,
        )
    except Exception:
        _load_state(model, original_base)
        raise
    finally:
        refreeze()


def _frozen_forward_batches(
    model: nn.Module, values: Tensor, *, batch_size: int
) -> tuple[Tensor, Tensor]:
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    logits: list[Tensor] = []
    features: list[Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            current_logits, current_features = _forward_identity(model, values[start : start + batch_size])
            logits.append(current_logits)
            features.append(current_features)
    return torch.cat(logits, dim=0), torch.cat(features, dim=0)


def predict_registered_logits(
    model: nn.Module,
    query_iq: Tensor,
    *,
    query_tokens: Sequence[str],
    class_registry: Sequence[str],
    batch_size: int = 128,
) -> Mapping[str, np.ndarray]:
    """Predict each opaque query independently over the complete registry."""

    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("MARC-OT query prediction requires a frozen eval model")
    values = torch.as_tensor(query_iq)
    tokens = tuple(query_tokens)
    registry = tuple(str(value) for value in class_registry)
    if values.ndim < 2 or values.shape[0] == 0 or not bool(torch.isfinite(values).all()):
        raise ValueError("query values must be finite nonempty rows")
    if len(tokens) != len(values) or len(set(tokens)) != len(tokens):
        raise ValueError("query tokens must be unique and align with query rows")
    if not registry or len(set(registry)) != len(registry):
        raise ValueError("class registry must be complete and unique")
    logits, _ = _frozen_forward_batches(model, values, batch_size=batch_size)
    if logits.shape != (len(values), len(registry)):
        raise ValueError("query logits do not face the complete registered class registry")
    predictions = logits.argmax(dim=1)
    return {
        "query_tokens": np.asarray(tokens),
        "logits": np.asarray(logits.detach().cpu().tolist(), dtype=np.float32),
        "predictions": np.asarray(predictions.detach().cpu().tolist(), dtype=np.int64),
        "class_registry": np.asarray(registry),
    }


def predict_marc_ot_probes(
    model: nn.Module,
    support_iq: Tensor,
    support_labels: Tensor,
    query_iq: Tensor,
    *,
    support_tokens: Sequence[str],
    query_tokens: Sequence[str],
    class_registry: Sequence[str],
    seed: int,
    batch_size: int = 128,
) -> Mapping[str, np.ndarray]:
    """Create P1/P2/P3 old-class predictions after support state is frozen."""

    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("MARC-OT query prediction requires a frozen eval model")
    support, labels, _ = _validate_support(support_iq, support_labels, support_tokens)
    query = torch.as_tensor(query_iq, device=support.device, dtype=support.dtype)
    registry = tuple(str(value) for value in class_registry)
    tokens = tuple(query_tokens)
    if len(registry) != int(torch.unique(labels).numel()) or len(tokens) != len(query):
        raise ValueError("prediction registry or query-token binding drift")
    support_logits, support_features = _frozen_forward_batches(
        model, support, batch_size=batch_size
    )
    query_logits, query_features = _frozen_forward_batches(model, query, batch_size=batch_size)
    if support_logits.shape[1] != len(registry) or query_logits.shape[1] != len(registry):
        raise ValueError("frozen source head registry drift")
    prototypes = torch.stack(
        [support_features[labels == class_id].mean(dim=0) for class_id in range(len(registry))]
    )
    p2_logits = functional.normalize(query_features.float(), dim=1) @ functional.normalize(
        prototypes.float(), dim=1
    ).T
    if support_features.shape[1] != 160 or support.ndim != 3 or tuple(support.shape[1:]) != (2, 256):
        raise ValueError("P3 old-only D92 requires 160D identity and [N,2,256] IQ")
    from .stage2_binova_d92 import exact_d92_fit
    from .stage2_binova_features import make_fft96

    support_np = np.asarray(support.detach().cpu().tolist(), dtype=np.float32)
    query_np = np.asarray(query.detach().cpu().tolist(), dtype=np.float32)
    support_identity = np.asarray(support_features.detach().cpu().tolist(), dtype=np.float32)
    query_identity = np.asarray(query_features.detach().cpu().tolist(), dtype=np.float32)
    labels_np = np.asarray(labels.detach().cpu().tolist(), dtype=np.int64)
    d92 = exact_d92_fit(
        support_identity,
        make_fft96(support_np),
        labels_np,
        class_ids=range(len(registry)),
        old_class_count=len(registry),
        seed=int(seed),
        device=str(query.device),
    )
    p3_logits = np.asarray(d92.score(query_identity, make_fft96(query_np)), dtype=np.float32)
    p3_predictions = np.asarray(d92.predict(query_identity, make_fft96(query_np)), dtype=np.int64)
    return {
        "query_tokens": np.asarray(tokens),
        "p1_logits": np.asarray(query_logits.detach().cpu().tolist(), dtype=np.float32),
        "p1_predictions": np.asarray(query_logits.argmax(dim=1).detach().cpu().tolist(), dtype=np.int64),
        "p2_logits": np.asarray(p2_logits.detach().cpu().tolist(), dtype=np.float32),
        "p2_predictions": np.asarray(p2_logits.argmax(dim=1).detach().cpu().tolist(), dtype=np.int64),
        "p3_logits": p3_logits,
        "p3_predictions": p3_predictions,
        "query_z_id": query_identity,
    }


__all__ = [
    "MARCOT_FORMAL_ARMS",
    "MARCOT_PROGRESSIVE_STAGES",
    "MARCOTRunnerConfig",
    "MARCOTTrainingAudit",
    "SupportSafeSelection",
    "combine_blockwise_gradients",
    "predict_marc_ot_probes",
    "predict_registered_logits",
    "resolve_block_learning_rates",
    "select_support_safe_state",
    "train_marc_ot_arm",
]
