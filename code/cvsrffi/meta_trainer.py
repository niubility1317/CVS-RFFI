"""Source-only Phase1 meta trainer and checkpoint selection.

This module is intentionally narrow.  Phase1-B and Phase1-C optimizers are
constructed from the actual Task3 modules, while the episode step delegates
the functional inner loop and pure objectives to Tasks 5 and 6.  Validation
is closed over typed carriers so target/query fields cannot be smuggled into
the source-only path through an open dictionary.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields, replace, is_dataclass
import math
import time
from typing import Any

import torch
from torch import Tensor, nn

from .meta_adapter import (
    ResidualMetaAdapter,
    adapter_parameter_budget,
    adapter_step_size_by_parameter,
    iter_inner_adapter_parameters,
)
from .meta_episodes import MetaEpisode
from .meta_inner_loop import (
    FastAdapterState,
    MetaInnerLoopError,
    first_order_adapt,
    functional_forward,
)
from .meta_objectives import (
    LossBreakdown,
    MetaObjectiveConfig,
    outer_objective,
    support_objective,
)


_PHASE1_CURVE_STEPS = (0, 1, 3, 5, 10)
_ADAPTER_INNER_SUFFIXES = (
    "down.weight",
    "down.bias",
    "up.weight",
    "up.bias",
    "gate",
)
_BACKBONE_MODULE_NAMES = frozenset({"t_proj", "f_proj", "fuse"})
_SOURCE_ROLES = frozenset({"L_s"})
_SOURCE_EVAL_ROLES = frozenset({"V_cal", "V_select"})


class MetaTrainerError(RuntimeError):
    """Raised when a source-only trainer contract cannot be proven."""


@dataclass(frozen=True)
class MetaTrainerConfig:
    """Finite Phase1-B/C trainer settings.

    ``inner_steps`` and ``meta_batch_size`` are explicit because they are
    part of the V1 experiment definition.  V1 training uses exactly three
    inner steps; the adaptation diagnostic has its own fixed step list.
    """

    adapter_outer_lr: float = 1.0e-3
    weight_decay: float = 0.0
    meta_batch_size: int = 4
    inner_steps: int = 3
    phase1c_backbone_lr_ratio: float = 0.05
    grad_clip_norm: float | None = 1.0
    objective_config: MetaObjectiveConfig = field(default_factory=MetaObjectiveConfig)

    def __post_init__(self) -> None:
        for name in ("adapter_outer_lr", "weight_decay", "phase1c_backbone_lr_ratio"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if float(self.adapter_outer_lr) <= 0.0:
            raise ValueError("adapter_outer_lr must be positive")
        if float(self.phase1c_backbone_lr_ratio) <= 0.0:
            raise ValueError("phase1c_backbone_lr_ratio must be positive")
        if isinstance(self.meta_batch_size, bool) or int(self.meta_batch_size) <= 0:
            raise ValueError("meta_batch_size must be a positive integer")
        if isinstance(self.inner_steps, bool) or int(self.inner_steps) != 3:
            raise ValueError("Phase1 V1 inner_steps must be exactly 3")
        if self.grad_clip_norm is not None:
            value = float(self.grad_clip_norm)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("grad_clip_norm must be finite and positive or None")
        if not isinstance(self.objective_config, MetaObjectiveConfig):
            raise TypeError("objective_config must be a MetaObjectiveConfig")
        object.__setattr__(self, "meta_batch_size", int(self.meta_batch_size))
        object.__setattr__(self, "inner_steps", int(self.inner_steps))

    @property
    def grad_clip(self) -> float | None:
        """Compatibility spelling for callers that use ``grad_clip``."""

        return self.grad_clip_norm


@dataclass(frozen=True)
class MetaEpisodeBatch:
    """Typed tensors paired with one Task2 source episode.

    Query rows are concatenated in the same order as
    ``episode.query_adapt + episode.query_guard``; boolean masks identify the
    two objective routes.  No target receiver, query-truth or Phase2 field is
    represented by this carrier.
    """

    episode: MetaEpisode
    support_x: Tensor
    support_y: Tensor
    query_x: Tensor
    query_y: Tensor
    adapt_mask: Tensor
    guard_mask: Tensor
    frozen_prototypes: Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.episode, MetaEpisode):
            raise TypeError("episode must be a Task2 MetaEpisode")
        for name in ("support_x", "query_x", "frozen_prototypes"):
            value = getattr(self, name)
            if not torch.is_tensor(value) or not value.is_floating_point():
                raise TypeError(f"{name} must be a floating-point tensor")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must contain only finite values")
        if self.support_x.ndim < 1 or self.query_x.ndim < 1:
            raise ValueError("support_x and query_x must have a batch dimension")
        if self.support_x.size(0) != len(self.episode.support):
            raise ValueError("support_x length must match episode.support")
        query_rows = self.episode.query_adapt + self.episode.query_guard
        if self.query_x.size(0) != len(query_rows):
            raise ValueError("query_x length must match episode query rows")
        for name, value, expected in (
            ("support_y", self.support_y, self.support_x.size(0)),
            ("query_y", self.query_y, self.query_x.size(0)),
        ):
            if not torch.is_tensor(value) or value.ndim != 1 or value.numel() != expected:
                raise ValueError(f"{name} must be a one-dimensional label tensor of length {expected}")
            if value.dtype not in (
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            ):
                raise ValueError(f"{name} must use an integer dtype")
        for name, value in (("adapt_mask", self.adapt_mask), ("guard_mask", self.guard_mask)):
            if not torch.is_tensor(value) or value.ndim != 1 or value.numel() != self.query_x.size(0):
                raise ValueError(f"{name} must match the query batch length")
            if value.dtype is not torch.bool:
                raise ValueError(f"{name} must use boolean dtype")
        if bool((self.adapt_mask & self.guard_mask).any()):
            raise ValueError("adapt_mask and guard_mask must not overlap")
        if (
            not torch.is_tensor(self.frozen_prototypes)
            or self.frozen_prototypes.ndim != 2
            or not self.frozen_prototypes.is_floating_point()
        ):
            raise ValueError("frozen_prototypes must be a floating tensor with shape [classes, dimension]")
        if not bool(torch.isfinite(self.frozen_prototypes).all()):
            raise ValueError("frozen_prototypes must contain only finite values")
        object.__setattr__(self, "frozen_prototypes", self.frozen_prototypes.detach())

        for index, row in enumerate(query_rows):
            if bool(self.adapt_mask[index]) and int(row.tx_i) not in self.episode.adapt_class_ids:
                raise ValueError("adapt_mask routes a query row outside adapt_class_ids")
            if bool(self.guard_mask[index]) and int(row.tx_i) not in self.episode.guard_class_ids:
                raise ValueError("guard_mask routes a query row outside guard_class_ids")
        support_labels = self.support_y.detach().cpu().tolist()
        if any(int(label) not in self.episode.adapt_class_ids for label in support_labels):
            raise ValueError("support labels must belong to adapt_class_ids")


@dataclass(frozen=True)
class MetaTrainStepResult:
    """Immutable result and audit of one meta batch outer update."""

    loss: Tensor
    episode_logs: tuple[Mapping[str, object], ...]
    optimizer_parameter_names: tuple[str, ...]
    updated_parameter_names: tuple[str, ...]
    parameter_audit: Mapping[str, object]

    @property
    def total_loss(self) -> Tensor:
        return self.loss

    @property
    def logs(self) -> tuple[Mapping[str, object], ...]:
        return self.episode_logs

    @property
    def optimizer_audit(self) -> Mapping[str, object]:
        return self.parameter_audit


@dataclass(frozen=True)
class AdaptationCurveRow:
    """One source holdout metric row at one fixed adaptation step."""

    episode_index: int
    role: str
    step: int
    episode_kind: str
    k_shot: int
    mean_accuracy: float | None
    floor_accuracy: float | None
    per_class_accuracy: tuple[tuple[int, float], ...]
    adapt_accuracy: float | None
    guard_accuracy: float | None
    clean_step0_accuracy: float | None
    adaptation_delta_pp: float | None
    held_receiver: tuple[int | str, ...]
    held_day: tuple[int, ...]
    held_channel: tuple[int, ...]
    leo_scenarios: tuple[str, ...]
    y_adapt_count: int
    y_guard_count: int
    adapter_norm: float
    module_step_sizes: tuple[tuple[str, float], ...]
    parameter_ratio: float
    state_size_bytes: int
    latency_ms: float
    source_only: bool = True


@dataclass(frozen=True)
class AdaptationCurve:
    """Fixed-step source-only adaptation curve."""

    steps: tuple[int, ...]
    rows: tuple[AdaptationCurveRow, ...]
    source_only: bool = True

    def __post_init__(self) -> None:
        if self.steps != _PHASE1_CURVE_STEPS:
            raise ValueError("Phase1 adaptation curve steps must be exactly (0, 1, 3, 5, 10)")
        if not self.source_only:
            raise ValueError("adaptation curve must be source-only")
        if any(row.step not in self.steps for row in self.rows):
            raise ValueError("adaptation curve row has an unsupported step")

    def __iter__(self):
        return iter(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, key):
        if isinstance(key, int) and key in self.steps:
            return tuple(row for row in self.rows if row.step == key)
        return self.rows[key]

    def rows_at(self, step: int) -> tuple[AdaptationCurveRow, ...]:
        if int(step) not in self.steps:
            raise KeyError(step)
        return tuple(row for row in self.rows if row.step == int(step))


@dataclass(frozen=True)
class SourceHoldoutDelta:
    """Typed source holdout A(3)-A(0) evidence."""

    holdout_id: str
    a0: float
    a3: float

    def __post_init__(self) -> None:
        if not str(self.holdout_id):
            raise ValueError("holdout_id must be non-empty")
        for name in ("a0", "a3"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")

    @property
    def delta_pp(self) -> float:
        return 100.0 * (float(self.a3) - float(self.a0))


@dataclass(frozen=True)
class SourceCheckpointCandidate:
    """Closed source-only candidate record used by the selector."""

    candidate_id: str
    clean_delta_pp: float
    guard_floor_delta_pp: float
    worst_a3_delta_pp: float | None = None
    parameter_count: int = 0
    latency_ms: float = 0.0
    source_holdout_deltas_pp: tuple[float, ...] = ()
    source_holdouts: tuple[SourceHoldoutDelta, ...] = ()

    def __post_init__(self) -> None:
        if not str(self.candidate_id):
            raise ValueError("candidate_id must be non-empty")
        for name in ("clean_delta_pp", "guard_floor_delta_pp", "latency_ms"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if name == "latency_ms" and value < 0.0:
                raise ValueError("latency_ms must be non-negative")
        if isinstance(self.parameter_count, bool) or int(self.parameter_count) < 0:
            raise ValueError("parameter_count must be a non-negative integer")
        deltas = tuple(float(value) for value in self.source_holdout_deltas_pp)
        if any(not math.isfinite(value) for value in deltas):
            raise ValueError("source_holdout_deltas_pp must be finite")
        holdouts = tuple(self.source_holdouts)
        if any(not isinstance(item, SourceHoldoutDelta) for item in holdouts):
            raise TypeError("source_holdouts must contain SourceHoldoutDelta values")
        derived = deltas + tuple(item.delta_pp for item in holdouts)
        worst = self.worst_a3_delta_pp
        if worst is None:
            if not derived:
                raise ValueError("candidate requires source holdout A(3)-A(0) evidence")
            worst = min(derived)
        worst = float(worst)
        if not math.isfinite(worst):
            raise ValueError("worst_a3_delta_pp must be finite")
        object.__setattr__(self, "candidate_id", str(self.candidate_id))
        object.__setattr__(self, "parameter_count", int(self.parameter_count))
        object.__setattr__(self, "source_holdout_deltas_pp", deltas)
        object.__setattr__(self, "source_holdouts", holdouts)
        object.__setattr__(self, "worst_a3_delta_pp", worst)

    @property
    def worst_source_holdout_delta_pp(self) -> float:
        return float(self.worst_a3_delta_pp)


def _module_parameters(model: nn.Module, module_names: frozenset[str]) -> list[tuple[str, nn.Parameter]]:
    """Collect parameters under exact module-name terminals, not substrings."""

    selected_ids: set[int] = set()
    for qualified_name, module in model.named_modules():
        if not qualified_name or qualified_name.rsplit(".", 1)[-1] not in module_names:
            continue
        if any(
            part.lower() in {"cls_head", "classifier", "prototype", "prototypes"}
            for part in qualified_name.split(".")
        ):
            continue
        selected_ids.update(id(parameter) for parameter in module.parameters(recurse=True))
    return [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if id(parameter) in selected_ids
    ]


def _adapter_outer_parameters(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    """Return Task3 inner leaves plus every owning module step size."""

    inner = OrderedDict(iter_inner_adapter_parameters(model))
    if not inner:
        raise ValueError("model has no enabled Task3 meta adapters")
    adapter_ids = {id(parameter) for parameter in inner.values()}
    step_ids: set[int] = set()
    for module_name, module in model.named_modules():
        if not isinstance(module, ResidualMetaAdapter):
            continue
        del module_name
        step_ids.add(id(module.log_step_size))
    if not step_ids:
        raise ValueError("model has no adapter log_step_size parameters")
    selected_ids = adapter_ids | step_ids
    return [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if id(parameter) in selected_ids
    ]


def _freeze_and_enable(model: nn.Module, selected: Sequence[tuple[str, nn.Parameter]]) -> None:
    selected_ids = {id(parameter) for _, parameter in selected}
    for parameter in model.parameters():
        parameter.requires_grad = id(parameter) in selected_ids


def _make_optimizer(
    model: nn.Module,
    config: MetaTrainerConfig,
    adapter_parameters: Sequence[tuple[str, nn.Parameter]],
    backbone_parameters: Sequence[tuple[str, nn.Parameter]],
) -> torch.optim.Optimizer:
    adapter_parameters = tuple(adapter_parameters)
    backbone_parameters = tuple(backbone_parameters)
    if not adapter_parameters:
        raise ValueError("optimizer requires at least one adapter parameter")
    selected = list(adapter_parameters) + list(backbone_parameters)
    _freeze_and_enable(model, selected)
    groups: list[dict[str, object]] = [
        {
            "params": [parameter for _, parameter in adapter_parameters],
            "lr": float(config.adapter_outer_lr),
            "weight_decay": float(config.weight_decay),
        }
    ]
    if backbone_parameters:
        groups.append(
            {
                "params": [parameter for _, parameter in backbone_parameters],
                "lr": float(config.adapter_outer_lr) * float(config.phase1c_backbone_lr_ratio),
                "weight_decay": float(config.weight_decay),
            }
        )
    return torch.optim.AdamW(groups)


def build_phase1b_optimizer(
    model: nn.Module,
    config: MetaTrainerConfig | None = None,
) -> torch.optim.Optimizer:
    """Build the Phase1-B adapter and module-step-size optimizer."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    config = MetaTrainerConfig() if config is None else config
    if not isinstance(config, MetaTrainerConfig):
        raise TypeError("config must be a MetaTrainerConfig")
    return _make_optimizer(model, config, _adapter_outer_parameters(model), ())


def build_phase1c_optimizer(
    model: nn.Module,
    config: MetaTrainerConfig | None = None,
) -> torch.optim.Optimizer:
    """Build Phase1-C with only exact ``t_proj/f_proj/fuse`` additions."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    config = MetaTrainerConfig() if config is None else config
    if not isinstance(config, MetaTrainerConfig):
        raise TypeError("config must be a MetaTrainerConfig")
    return _make_optimizer(
        model,
        config,
        _adapter_outer_parameters(model),
        _module_parameters(model, _BACKBONE_MODULE_NAMES),
    )


def optimizer_parameter_names(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    group: Mapping[str, object] | None = None,
) -> tuple[str, ...]:
    """Resolve optimizer tensors back to exact model parameter names."""

    if not isinstance(model, nn.Module) or not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("model and optimizer types are invalid")
    named = {id(parameter): name for name, parameter in model.named_parameters()}
    groups = (group,) if group is not None else tuple(optimizer.param_groups)
    parameters: list[nn.Parameter] = []
    for current_group in groups:
        if not isinstance(current_group, Mapping) or "params" not in current_group:
            raise ValueError("optimizer group must contain params")
        parameters.extend(current_group["params"])  # type: ignore[arg-type]
    resolved: list[str] = []
    for parameter in parameters:
        name = named.get(id(parameter))
        if name is None:
            raise ValueError("optimizer contains a parameter outside model")
        resolved.append(name)
    return tuple(resolved)


def _forbidden_metadata(value: Any, *, path: str = "") -> str | None:
    """Find explicit target/query leakage markers in typed episode metadata."""

    if isinstance(value, str):
        lowered = value.lower()
        if any(token in lowered for token in ("target", "phase2", "query_truth", "query-role")):
            return f"{path}={value!r}"
        return None
    if is_dataclass(value) and not isinstance(value, type):
        for item in fields(value):
            marker = _forbidden_metadata(
                getattr(value, item.name),
                path=f"{path}.{item.name}",
            )
            if marker:
                return marker
        return None
    if isinstance(value, Mapping):
        for key, item in value.items():
            marker = _forbidden_metadata(str(key), path=f"{path}.key")
            if marker:
                return marker
            marker = _forbidden_metadata(item, path=f"{path}.{key}")
            if marker:
                return marker
        return None
    if isinstance(value, (tuple, list, set, frozenset)):
        for index, item in enumerate(value):
            marker = _forbidden_metadata(item, path=f"{path}[{index}]")
            if marker:
                return marker
        return None
    return None


def _episode_rows(episode: MetaEpisode):
    return episode.support + episode.query_adapt + episode.query_guard


def _validate_episode_roles(batch: MetaEpisodeBatch, allowed: frozenset[str]) -> None:
    marker = _forbidden_metadata(_episode_rows(batch.episode), path="episode")
    if marker:
        raise ValueError(f"source-only episode contains forbidden target/query field: {marker}")
    rows = _episode_rows(batch.episode)
    roles = {str(row.role) for row in rows}
    if not roles.issubset(allowed):
        expected = ", ".join(sorted(allowed))
        raise ValueError(f"episode roles must be restricted to {expected}; got {sorted(roles)!r}")


def _initial_fast_state(model: nn.Module) -> FastAdapterState:
    values = OrderedDict((name, parameter.clone()) for name, parameter in iter_inner_adapter_parameters(model))
    if not values:
        raise MetaInnerLoopError("model has no Task3 inner adapter parameters")
    return FastAdapterState(values, 0, ())


def _support_loss_fn_for_model(
    model: nn.Module,
    batch: MetaEpisodeBatch,
    config: MetaTrainerConfig,
):
    initial = OrderedDict((name, parameter) for name, parameter in iter_inner_adapter_parameters(model))

    def loss_fn(outputs: Mapping[str, object], labels: Tensor, current: Mapping[str, Tensor]) -> Tensor:
        return support_objective(
            outputs,
            labels,
            batch.frozen_prototypes,
            initial,
            current,
            config.objective_config,
        ).total

    return loss_fn


def _extract_logits(outputs: Mapping[str, object]) -> Tensor:
    if not isinstance(outputs, Mapping):
        raise ValueError("model return_aux output must be a mapping")
    values = []
    for key in ("logits", "tx_logits"):
        if key in outputs:
            value = outputs[key]
            if not torch.is_tensor(value):
                raise ValueError(f"{key} must be a tensor")
            values.append((key, value))
    if not values:
        raise ValueError("model output is missing fixed-head logits")
    first_key, logits = values[0]
    for key, value in values[1:]:
        if value.shape != logits.shape or value.dtype != logits.dtype or value.device != logits.device:
            raise ValueError(f"logit aliases {first_key} and {key} conflict")
        if not torch.equal(value, logits):
            raise ValueError(f"logit aliases {first_key} and {key} conflict")
    if logits.ndim != 2 or not logits.is_floating_point() or not bool(torch.isfinite(logits).all()):
        raise ValueError("logits must be a finite floating [batch, classes] tensor")
    return logits


def _cosine_or_none(
    left: Sequence[Tensor | None],
    right: Sequence[Tensor | None],
) -> float | None:
    if len(left) != len(right) or any(a is None or b is None for a, b in zip(left, right)):
        return None
    left_values = [value.detach().reshape(-1) for value in left if value is not None]
    right_values = [value.detach().reshape(-1) for value in right if value is not None]
    if not left_values:
        return None
    left_flat = torch.cat(left_values)
    right_flat = torch.cat(right_values)
    if not bool(torch.isfinite(left_flat).all()) or not bool(torch.isfinite(right_flat).all()):
        return None
    left_norm = torch.linalg.vector_norm(left_flat)
    right_norm = torch.linalg.vector_norm(right_flat)
    if float(left_norm.detach().cpu()) == 0.0 or float(right_norm.detach().cpu()) == 0.0:
        return 0.0
    value = torch.dot(left_flat, right_flat) / (left_norm * right_norm)
    return float(value.detach().cpu().item()) if bool(torch.isfinite(value)) else None


def _validate_optimizer_scope(model: nn.Module, optimizer: torch.optim.Optimizer) -> tuple[str, ...]:
    names = optimizer_parameter_names(model, optimizer)
    if not names:
        raise ValueError("optimizer must contain parameters")
    inner_names = {name for name, _ in iter_inner_adapter_parameters(model)}
    log_names = {
        name
        for name, parameter in model.named_parameters()
        if name.endswith("log_step_size")
        and any(id(parameter) == id(module.log_step_size) for module in model.modules() if isinstance(module, ResidualMetaAdapter))
    }
    backbone_names = {name for name, _ in _module_parameters(model, _BACKBONE_MODULE_NAMES)}
    allowed = inner_names | log_names | backbone_names
    if set(names) - allowed:
        raise ValueError(
            "optimizer contains parameters outside Phase1-B/C whitelist: "
            f"{sorted(set(names) - allowed)!r}"
        )
    if not inner_names.issubset(names) or not log_names.issubset(names):
        raise ValueError("optimizer must include every adapter inner and log_step_size parameter")
    return names


def _snapshot_state(model: nn.Module):
    parameters = {name: (parameter, parameter.detach().clone()) for name, parameter in model.named_parameters()}
    buffers = {name: (buffer, buffer.detach().clone()) for name, buffer in model.named_buffers() if buffer is not None}
    return parameters, buffers


def _restore_state(snapshot) -> None:
    parameters, buffers = snapshot
    with torch.no_grad():
        for parameter, before in parameters.values():
            parameter.copy_(before)
        for buffer, before in buffers.values():
            buffer.copy_(before)


def _state_changed_names(model: nn.Module, snapshot) -> set[str]:
    parameters, buffers = snapshot
    changed: set[str] = set()
    for name, (parameter, before) in parameters.items():
        if not torch.equal(parameter.detach(), before):
            changed.add(name)
    for name, (buffer, before) in buffers.items():
        if not torch.equal(buffer.detach(), before):
            changed.add(name)
    return changed


def run_meta_train_step(
    model: nn.Module,
    episodes: Sequence[MetaEpisodeBatch],
    optimizer: torch.optim.Optimizer,
    config: MetaTrainerConfig | None = None,
) -> MetaTrainStepResult:
    """Run one source-only FOMAML meta batch and exactly one outer step."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch.optim.Optimizer")
    config = MetaTrainerConfig() if config is None else config
    if not isinstance(config, MetaTrainerConfig):
        raise TypeError("config must be a MetaTrainerConfig")
    if isinstance(episodes, MetaEpisodeBatch):
        episodes = (episodes,)
    else:
        episodes = tuple(episodes)
    if len(episodes) != config.meta_batch_size:
        raise ValueError(
            f"meta batch must contain exactly {config.meta_batch_size} episodes; got {len(episodes)}"
        )
    for batch in episodes:
        if not isinstance(batch, MetaEpisodeBatch):
            raise TypeError("run_meta_train_step accepts MetaEpisodeBatch values only")
        _validate_episode_roles(batch, _SOURCE_ROLES)
    optimizer_names = _validate_optimizer_scope(model, optimizer)
    snapshot = _snapshot_state(model)
    optimizer.zero_grad(set_to_none=True)
    outer_losses: list[Tensor] = []
    logs: list[Mapping[str, object]] = []
    try:
        for batch in episodes:
            initial_state = _initial_fast_state(model)
            support_loss_fn = _support_loss_fn_for_model(model, batch, config)
            fast_state = first_order_adapt(
                model,
                batch.support_x,
                batch.support_y,
                support_loss_fn,
                steps=config.inner_steps,
            )
            pre_outputs = functional_forward(model, initial_state, batch.query_x, batch.query_y)
            post_outputs = functional_forward(model, fast_state, batch.query_x, batch.query_y)
            losses: LossBreakdown = outer_objective(
                pre_outputs,
                post_outputs,
                batch.query_y,
                batch.adapt_mask,
                batch.guard_mask,
                batch.frozen_prototypes,
                config.objective_config,
            )
            if not torch.is_tensor(losses.total) or losses.total.ndim != 0 or not bool(torch.isfinite(losses.total)):
                raise MetaTrainerError("outer loss must be a finite scalar")

            support_outputs = functional_forward(
                model,
                initial_state,
                batch.support_x,
                batch.support_y,
            )
            support_breakdown = support_objective(
                support_outputs,
                batch.support_y,
                batch.frozen_prototypes,
                OrderedDict((name, parameter) for name, parameter in iter_inner_adapter_parameters(model)),
                initial_state.parameters,
                config.objective_config,
            )
            try:
                support_grads = torch.autograd.grad(
                    support_breakdown.total,
                    tuple(initial_state.parameters.values()),
                    allow_unused=True,
                    retain_graph=False,
                )
                query_grads = torch.autograd.grad(
                    losses.total,
                    tuple(fast_state.parameters.values()),
                    allow_unused=True,
                    retain_graph=True,
                )
            except RuntimeError:
                support_grads = tuple()
                query_grads = tuple()
            outer_losses.append(losses.total)
            grad_cos = _cosine_or_none(support_grads, query_grads)
            logs.append(
                {
                    "episode_kind": batch.episode.kind.value,
                    "k_shot": int(batch.episode.k_shot),
                    "inner_steps": int(fast_state.steps),
                    "loss_adapt": float(losses.adapt.detach().cpu().item()),
                    "loss_guard": float(losses.guard.detach().cpu().item()),
                    "loss_floor": float(losses.floor.detach().cpu().item()),
                    "grad_cos_support_query": grad_cos,
                }
            )
        total_loss = torch.stack(outer_losses).mean()
        if not bool(torch.isfinite(total_loss)) or not total_loss.requires_grad:
            raise MetaTrainerError("meta batch outer loss is non-finite or detached")
        total_loss.backward()
        trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        for name, parameter in model.named_parameters():
            if name in optimizer_names and parameter.grad is not None:
                if not bool(torch.isfinite(parameter.grad).all()):
                    raise MetaTrainerError(f"non-finite outer gradient for {name!r}")
        if config.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(trainable_parameters, float(config.grad_clip_norm), error_if_nonfinite=True)
        optimizer.step()
        changed = _state_changed_names(model, snapshot)
        if not changed.issubset(set(optimizer_names)):
            _restore_state(snapshot)
            raise MetaTrainerError(
                "outer step changed a parameter or buffer outside optimizer whitelist: "
                f"{sorted(changed - set(optimizer_names))!r}"
            )
        audit = {
            "optimizer_parameter_names": tuple(optimizer_names),
            "updated_parameter_names": tuple(sorted(changed)),
            "non_optimizer_state_unchanged": True,
        }
        return MetaTrainStepResult(
            loss=total_loss.detach(),
            episode_logs=tuple(logs),
            optimizer_parameter_names=tuple(optimizer_names),
            updated_parameter_names=tuple(sorted(changed)),
            parameter_audit=audit,
        )
    except Exception:
        _restore_state(snapshot)
        raise
    finally:
        for parameter in model.parameters():
            parameter.grad = None


def _accuracy_rows(logits: Tensor, labels: Tensor, mask: Tensor) -> tuple[float | None, float | None, tuple[tuple[int, float], ...]]:
    selected = mask.to(device=logits.device) & torch.ones_like(mask, dtype=torch.bool, device=logits.device)
    if not bool(selected.any()):
        return None, None, ()
    predictions = logits.argmax(dim=1)
    labels = labels.to(device=logits.device, dtype=torch.long)
    selected_labels = labels[selected]
    correct = (predictions[selected] == selected_labels)
    pairs: list[tuple[int, float]] = []
    for class_id in torch.unique(selected_labels, sorted=True).tolist():
        class_mask = selected_labels == int(class_id)
        pairs.append((int(class_id), float(correct[class_mask].float().mean().item())))
    values = [value for _, value in pairs]
    return float(correct.float().mean().item()), min(values), tuple(pairs)


def _eval_role(batch: MetaEpisodeBatch) -> str:
    roles = sorted({str(row.role) for row in _episode_rows(batch.episode)})
    return "+".join(roles)


def _validate_source_eval_batch(batch: MetaEpisodeBatch) -> None:
    if not isinstance(batch, MetaEpisodeBatch):
        raise TypeError("evaluate_adaptation_curve accepts MetaEpisodeBatch values only")
    _validate_episode_roles(batch, _SOURCE_EVAL_ROLES)
    rows = _episode_rows(batch.episode)
    if any(isinstance(row.rx_i, str) and "target" in row.rx_i.lower() for row in rows):
        raise ValueError("target receiver identifiers are forbidden in source adaptation curves")
    if any(str(row.role).lower().startswith("target") for row in rows):
        raise ValueError("target roles are forbidden in source adaptation curves")


def _curve_row(
    *,
    model: nn.Module,
    batch: MetaEpisodeBatch,
    episode_index: int,
    step: int,
    logits: Tensor,
    clean_step0_accuracy: float | None,
    fast_state: FastAdapterState,
    elapsed_ms: float,
) -> AdaptationCurveRow:
    query_mask = batch.adapt_mask | batch.guard_mask
    mean_accuracy, floor_accuracy, per_class = _accuracy_rows(logits, batch.query_y, query_mask)
    adapt_accuracy, _, _ = _accuracy_rows(logits, batch.query_y, batch.adapt_mask)
    guard_accuracy, _, _ = _accuracy_rows(logits, batch.query_y, batch.guard_mask)
    refs = batch.episode.query_adapt + batch.episode.query_guard
    receiver_ids = tuple(sorted({row.rx_i for row in refs}, key=str))
    day_ids = tuple(sorted({int(row.day_i) for row in refs}))
    channel_ids = tuple(sorted({int(row.capture_block_i) for row in refs}))
    leo_scenarios = tuple(
        sorted({str(row.view) for row in refs if str(row.view).startswith("leo_") and str(row.view).endswith("_weak")})
    )
    adapter_norm = torch.sqrt(
        sum((value.detach().pow(2).sum() for value in fast_state.parameters.values()), torch.zeros((), device=logits.device))
    )
    module_step_sizes: list[tuple[str, float]] = []
    step_map = adapter_step_size_by_parameter(model)
    for module_name, module in model.named_modules():
        if isinstance(module, ResidualMetaAdapter):
            value = module.step_size().detach()
            if not bool(torch.isfinite(value)):
                raise MetaTrainerError("adapter module step size is non-finite")
            module_step_sizes.append((module_name, float(value.cpu().item())))
    state_size = int(sum(value.numel() * value.element_size() for value in fast_state.parameters.values()))
    del step_map
    return AdaptationCurveRow(
        episode_index=int(episode_index),
        role=_eval_role(batch),
        step=int(step),
        episode_kind=batch.episode.kind.value,
        k_shot=int(batch.episode.k_shot),
        mean_accuracy=mean_accuracy,
        floor_accuracy=floor_accuracy,
        per_class_accuracy=per_class,
        adapt_accuracy=adapt_accuracy,
        guard_accuracy=guard_accuracy,
        clean_step0_accuracy=clean_step0_accuracy,
        adaptation_delta_pp=None,
        held_receiver=receiver_ids,
        held_day=day_ids,
        held_channel=channel_ids,
        leo_scenarios=leo_scenarios,
        y_adapt_count=int(batch.adapt_mask.sum().item()),
        y_guard_count=int(batch.guard_mask.sum().item()),
        adapter_norm=float(adapter_norm.cpu().item()),
        module_step_sizes=tuple(module_step_sizes),
        parameter_ratio=float(adapter_parameter_budget(model)["inner_ratio"]),
        state_size_bytes=state_size,
        latency_ms=float(elapsed_ms),
    )


def evaluate_adaptation_curve(
    model: nn.Module,
    episodes: Sequence[MetaEpisodeBatch],
    config: MetaTrainerConfig | None = None,
) -> AdaptationCurve:
    """Evaluate source ``V_cal/V_select`` at exactly A(0/1/3/5/10)."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    config = MetaTrainerConfig() if config is None else config
    if not isinstance(config, MetaTrainerConfig):
        raise TypeError("config must be a MetaTrainerConfig")
    if isinstance(episodes, MetaEpisodeBatch):
        episodes = (episodes,)
    else:
        episodes = tuple(episodes)
    if not episodes:
        raise ValueError("source adaptation curve requires at least one episode")
    for batch in episodes:
        _validate_source_eval_batch(batch)

    for parameter in model.parameters():
        parameter.grad = None
    snapshot = _snapshot_state(model)
    rows: list[AdaptationCurveRow] = []
    try:
        for episode_index, batch in enumerate(episodes):
            initial_state = _initial_fast_state(model)
            baseline_clean: float | None = None
            episode_rows: list[AdaptationCurveRow] = []
            for step in _PHASE1_CURVE_STEPS:
                started = time.perf_counter()
                if step == 0:
                    fast_state = initial_state
                else:
                    fast_state = first_order_adapt(
                        model,
                        batch.support_x,
                        batch.support_y,
                        _support_loss_fn_for_model(model, batch, config),
                        steps=step,
                    )
                with torch.no_grad():
                    outputs = functional_forward(model, fast_state, batch.query_x, batch.query_y)
                    logits = _extract_logits(outputs)
                    query_mask = batch.adapt_mask | batch.guard_mask
                    clean_mask = torch.tensor(
                        [str(row.view) == "clean" for row in batch.episode.query_adapt + batch.episode.query_guard],
                        dtype=torch.bool,
                        device=logits.device,
                    ) & query_mask
                    clean_accuracy, _, _ = _accuracy_rows(logits, batch.query_y, clean_mask)
                    if step == 0:
                        baseline_clean = clean_accuracy
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                row = _curve_row(
                    model=model,
                    batch=batch,
                    episode_index=episode_index,
                    step=step,
                    logits=logits,
                    clean_step0_accuracy=baseline_clean,
                    fast_state=fast_state,
                    elapsed_ms=elapsed_ms,
                )
                episode_rows.append(row)
            baseline = next((row.mean_accuracy for row in episode_rows if row.step == 0), None)
            for row in episode_rows:
                delta = None if baseline is None or row.mean_accuracy is None else 100.0 * (row.mean_accuracy - baseline)
                rows.append(replace(row, adaptation_delta_pp=delta))
    finally:
        _restore_state(snapshot)
        for parameter in model.parameters():
            parameter.grad = None
    changed = _state_changed_names(model, snapshot)
    if changed:
        raise MetaTrainerError(f"source curve changed model state: {sorted(changed)!r}")
    return AdaptationCurve(steps=_PHASE1_CURVE_STEPS, rows=tuple(rows))


def select_source_checkpoint(
    candidates: Sequence[SourceCheckpointCandidate],
) -> SourceCheckpointCandidate:
    """Select a source checkpoint under zero-step, floor and worst-holdout rules."""

    if isinstance(candidates, (str, bytes)):
        raise TypeError("candidates must be a sequence of typed source candidates")
    candidates = tuple(candidates)
    if not candidates:
        raise ValueError("source checkpoint candidate set is empty")
    for candidate in candidates:
        if not isinstance(candidate, SourceCheckpointCandidate):
            raise TypeError("source selection accepts SourceCheckpointCandidate values only")
        marker = _forbidden_metadata(candidate)
        if marker:
            raise ValueError(f"source checkpoint candidate contains forbidden target/query field: {marker}")
    eligible = [
        candidate
        for candidate in candidates
        if float(candidate.clean_delta_pp) >= -0.5
        and float(candidate.guard_floor_delta_pp) >= 0.0
    ]
    if not eligible:
        raise ValueError("no eligible source checkpoint candidate remains")
    eligible.sort(
        key=lambda candidate: (
            -float(candidate.worst_a3_delta_pp),
            int(candidate.parameter_count),
            float(candidate.latency_ms),
            str(candidate.candidate_id),
        )
    )
    return eligible[0]


__all__ = [
    "AdaptationCurve",
    "AdaptationCurveRow",
    "MetaEpisodeBatch",
    "MetaTrainStepResult",
    "MetaTrainerConfig",
    "MetaTrainerError",
    "SourceCheckpointCandidate",
    "SourceHoldoutDelta",
    "build_phase1b_optimizer",
    "build_phase1c_optimizer",
    "evaluate_adaptation_curve",
    "optimizer_parameter_names",
    "run_meta_train_step",
    "select_source_checkpoint",
]
