"""Stage-controlled Phase1 HCF-DG training.

The trainer owns only the source-domain optimization loop.  It deliberately
keeps the data boundary narrow: Stage0 receives IQ and environment metadata,
while identity labels are read only from the labelled source loader during
the main stages.  Target/query/truth inputs are rejected at construction.

The model and loss modules are developed independently, so the small adapter
surface below accepts the planned ``HCFDGModel`` call signature as well as a
model-provided ``loss``/``stage0_step``.  No Phase2 object is imported here.
"""

from __future__ import annotations

import csv
import inspect
import json
import math
import time
from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn
from torch.nn import functional as F

from .config import (
    HCFDGConfig,
    StageBudget,
    V1_AMP_ENABLED,
    V1_BACKBONE_LR,
    V1_COSFACE_FINAL_MARGIN,
    V1_COSFACE_MARGIN_RAMP_FRACTION,
    V1_COSINE_MIN_LR,
    V1_HEAD_LR,
    V1_WARMUP_FRACTION,
    V2_OPTIMIZER_UPDATES,
    V2_STAGE_BUDGET,
)


def warmup_updates(total_updates: int, fraction: float = V1_WARMUP_FRACTION) -> int:
    """Return the exact number of one-indexed warm-up updates."""

    total = int(total_updates)
    if total <= 0:
        raise ValueError("total_updates must be positive")
    if not 0.0 < float(fraction) < 1.0:
        raise ValueError("warm-up fraction must be between zero and one")
    return max(1, math.ceil(total * float(fraction)))


def learning_rate_at(
    update: int,
    *,
    total_updates: int,
    base_lr: float,
    min_lr: float = V1_COSINE_MIN_LR,
    warmup_fraction: float = V1_WARMUP_FRACTION,
) -> float:
    """Return the one-indexed linear-warm-up/cosine-decay learning rate.

    Update zero is the pre-step value.  At the warm-up boundary the rate is
    exactly ``base_lr``; at the final update it is exactly ``min_lr``.
    """

    total = int(total_updates)
    if total <= 0:
        raise ValueError("total_updates must be positive")
    if float(base_lr) < 0.0 or float(min_lr) < 0.0:
        raise ValueError("learning rates must be non-negative")
    if float(base_lr) < float(min_lr):
        raise ValueError("base_lr must not be below min_lr")

    current = max(0, min(int(update), total))
    if current == 0:
        return 0.0

    warmup = warmup_updates(total, warmup_fraction)
    if current <= warmup:
        return float(base_lr) * (current / warmup)

    decay_updates = total - warmup
    progress = (current - warmup) / decay_updates
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_lr) + (float(base_lr) - float(min_lr)) * cosine


def cosface_margin_at(
    update: int,
    *,
    total_updates: int,
    final_margin: float = V1_COSFACE_FINAL_MARGIN,
    ramp_fraction: float = V1_COSFACE_MARGIN_RAMP_FRACTION,
) -> float:
    """Return the one-indexed linear CosFace margin ramp."""

    total = int(total_updates)
    if total <= 0:
        raise ValueError("total_updates must be positive")
    if float(final_margin) < 0.0:
        raise ValueError("final_margin must be non-negative")
    if not 0.0 < float(ramp_fraction) <= 1.0:
        raise ValueError("margin ramp fraction must be in (0, 1]")

    current = max(0, min(int(update), total))
    if current == 0:
        return 0.0
    ramp_updates = max(1, math.ceil(total * float(ramp_fraction)))
    return float(final_margin) * min(1.0, current / ramp_updates)


def stage_for_update(update: int, budget: StageBudget = V2_STAGE_BUDGET) -> str:
    """Map a one-indexed V2 update to its frozen Stage0–4 interval."""

    current = int(update)
    if current < 1 or current > budget.total_updates:
        raise ValueError(f"update must be in [1, {budget.total_updates}]")
    boundaries = (
        ("stage0", budget.stage0),
        ("stage1", budget.stage0 + budget.stage1),
        ("stage2", budget.stage0 + budget.stage1 + budget.stage2),
        ("stage3", budget.stage0 + budget.stage1 + budget.stage2 + budget.stage3),
        ("stage4", budget.total_updates),
    )
    for name, end in boundaries:
        if current <= end:
            return name
    raise AssertionError("unreachable stage boundary")


@dataclass
class CheckpointPayload:
    """Serializable checkpoint contract for a frozen HCF-DG candidate."""

    phase1_method: str
    candidate_id: str
    source_split: Any
    fold: Any
    seed: Any
    update: int
    config: Mapping[str, Any]
    model_state: Mapping[str, Any]
    optimizer_state: Mapping[str, Any]
    scaler_state: Mapping[str, Any]
    inference: Mapping[str, Any]

    @property
    def model_state_dict(self) -> Mapping[str, Any]:
        return self.model_state

    @property
    def optimizer_state_dict(self) -> Mapping[str, Any]:
        return self.optimizer_state

    @property
    def scaler_state_dict(self) -> Mapping[str, Any]:
        return self.scaler_state

    @property
    def inference_metadata(self) -> Mapping[str, Any]:
        return self.inference

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical torch-save/JSON-inspection representation."""

        return {
            "phase1_method": self.phase1_method,
            "candidate_id": self.candidate_id,
            "source_split": self.source_split,
            "fold": self.fold,
            "seed": self.seed,
            "update": self.update,
            "config": dict(self.config),
            "model_state": self.model_state,
            "model": self.model_state,
            "optimizer_state": self.optimizer_state,
            "optimizer": self.optimizer_state,
            "scaler_state": self.scaler_state,
            "scaler": self.scaler_state,
            "inference": dict(self.inference),
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def keys(self):
        return self.to_dict().keys()


@dataclass
class TrainState:
    """Observable state returned by :meth:`HCFDGTrainer.train`."""

    optimizer_updates: int = 0
    backbone_forward_calls: int = 0
    stage_updates: dict[str, int] = field(default_factory=dict)
    freeze_update: int | None = None
    environment_updates: int = 0
    amp_enabled: bool = False
    metrics: list[dict[str, Any]] = field(default_factory=list)
    checkpoint: CheckpointPayload | None = None
    elapsed_seconds: float = 0.0
    total_gpu_hours: float = 0.0

    @property
    def main_optimizer_updates(self) -> int:
        """Number of identity-main updates, excluding V2 Stage0."""

        return self.optimizer_updates - self.stage_updates.get("stage0", 0)

    def to_dict(self) -> dict[str, Any]:
        result = {
            "optimizer_updates": self.optimizer_updates,
            "backbone_forward_calls": self.backbone_forward_calls,
            "stage_updates": dict(self.stage_updates),
            "freeze_update": self.freeze_update,
            "environment_updates": self.environment_updates,
            "amp_enabled": self.amp_enabled,
            "elapsed_seconds": self.elapsed_seconds,
            "total_gpu_hours": self.total_gpu_hours,
        }
        return result


@dataclass
class _SourceBatch:
    iq: Any = None
    receiver: Any = None
    day: Any = None
    channel: Any = None
    tx: Any = None
    domain: Any = None
    query_domain: Any = None
    support_mask: Any = None
    query_mask: Any = None
    content_keys: Any = None
    groups: Any = None
    env_meta: dict[str, Any] = field(default_factory=dict)


class _LoaderCursor:
    """Restart a finite loader without caching or peeking at its records."""

    def __init__(self, loader: Iterable[Any] | Callable[[], Iterable[Any]] | None):
        self.loader = loader
        self.iterator: Any = None

    def _new_iterator(self):
        if self.loader is None:
            return None
        source = self.loader() if callable(self.loader) and not isinstance(self.loader, Mapping) else self.loader
        return iter(source)

    def next(self) -> Any:
        if self.loader is None:
            return None
        if self.iterator is None:
            self.iterator = self._new_iterator()
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = self._new_iterator()
            if self.iterator is None:
                return None
            try:
                return next(self.iterator)
            except StopIteration:
                return None


def _invoke_supported(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call a planned interface while tolerating future optional arguments."""

    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(*args, **kwargs)

    parameters = signature.parameters.values()
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return function(*args, **kwargs)
    accepted = {parameter.name for parameter in parameters}
    filtered = {key: value for key, value in kwargs.items() if key in accepted}
    return function(*args, **filtered)


def _invoke_loss_function(function: Callable[..., Any], output: Any, **kwargs: Any) -> Any:
    """Pass the model output once to positional or keyword-only loss APIs."""

    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(output, **kwargs)
    has_positional = any(
        parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for parameter in signature.parameters.values()
    )
    if has_positional:
        return _invoke_supported(function, output, **kwargs)
    return _invoke_supported(function, output=output, **kwargs)


def _mapping_value(batch: Any, names: tuple[str, ...]) -> Any:
    """Read only one of the explicitly permitted source fields."""

    if isinstance(batch, Mapping):
        for name in names:
            try:
                return batch[name]
            except KeyError:
                continue
        return None
    for name in names:
        try:
            return getattr(batch, name)
        except AttributeError:
            continue
    return None


def _mapping_metadata(batch: Any) -> Any:
    return _mapping_value(batch, ("env_meta", "environment", "metadata", "meta"))


def _source_batch(batch: Any, *, allow_tx: bool, stage0: bool = False) -> _SourceBatch:
    """Extract IQ and source environment fields without opening query fields."""

    if torch.is_tensor(batch):
        return _SourceBatch(iq=batch)
    if isinstance(batch, (tuple, list)):
        iq = batch[0] if batch else None
        tx = batch[1] if allow_tx and len(batch) > 1 else None
        metadata = batch[2] if len(batch) > 2 else None
    else:
        iq = _mapping_value(batch, ("iq", "x", "inputs", "signal", "IQ"))
        tx = (
            _mapping_value(batch, ("tx", "tx_labels", "tx_ids", "label", "labels", "y"))
            if allow_tx
            else None
        )
        metadata = _mapping_metadata(batch)

    def read(name: str, aliases: tuple[str, ...]) -> Any:
        value = _mapping_value(batch, aliases)
        if value is not None:
            return value
        return _mapping_value(metadata, aliases) if metadata is not None else None

    receiver = read("receiver", ("receiver", "receiver_id", "receiver_ids", "rx", "rx_id", "rx_ids"))
    day = read("day", ("day", "day_id", "day_ids"))
    channel = None if stage0 else read("channel", ("channel", "channel_id", "channel_ids", "scenario"))
    env_meta: dict[str, Any] = {}
    if receiver is not None:
        env_meta["receiver"] = receiver
    if day is not None:
        env_meta["day"] = day
    if channel is not None:
        env_meta["channel"] = channel

    q_phys = None if stage0 else read("q_phys", ("q_phys", "physical_stats", "phys_stats"))
    if q_phys is not None:
        env_meta["q_phys"] = q_phys
    if not stage0:
        episode_type = read("episode_type", ("episode_type",))
        valid_tx_mask = read("valid_tx_mask", ("valid_tx_mask",))
        if episode_type is not None:
            env_meta["episode_type"] = episode_type
        if valid_tx_mask is not None:
            env_meta["valid_tx_mask"] = valid_tx_mask

    if stage0:
        domain = query_domain = support_mask = query_mask = content_keys = groups = None
    else:
        domain = read("domain", ("domain", "domain_id", "domain_ids"))
        query_domain = read("query_domain", ("query_domain", "heldout_domain"))
        support_mask = read("support_mask", ("support_mask",))
        query_mask = read("query_mask", ("query_mask",))
        content_keys = read("content_keys", ("content_keys", "content_key"))
        groups = read("groups", ("groups", "group_masks"))

    return _SourceBatch(
        iq=iq,
        receiver=receiver,
        day=day,
        channel=channel,
        tx=tx,
        domain=domain,
        query_domain=query_domain,
        support_mask=support_mask,
        query_mask=query_mask,
        content_keys=content_keys,
        groups=groups,
        env_meta=env_meta,
    )


def _move_value(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _move_value(item, device) for key, item in value.items()}
    return value


def _as_loss(value: Any) -> Any:
    if isinstance(value, Mapping):
        for key in ("loss", "total_loss", "objective", "total"):
            if key in value:
                return value[key]
    if isinstance(value, (tuple, list)) and value:
        return value[0]
    for name in ("loss", "total_loss", "objective", "total"):
        if hasattr(value, name):
            return getattr(value, name)
    return value


def _object_value(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _cheap_environment_inputs(iq: torch.Tensor, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build detached, IQ-only environment inputs without a backbone call."""

    if width <= 0:
        raise ValueError("environment input width must be positive")
    values = iq.abs() if torch.is_complex(iq) else iq
    values = values.float()
    if values.ndim == 0:
        values = values.reshape(1, 1)
    batch_size = int(values.shape[0])
    flattened = values.reshape(batch_size, -1)
    features = F.adaptive_avg_pool1d(flattened.unsqueeze(1), width).squeeze(1)
    q_phys = flattened.square().mean(dim=1, keepdim=True).sqrt()
    return features, q_phys


def _safe_environment_cross_entropy(logits: Any, target: Any) -> torch.Tensor | None:
    if not torch.is_tensor(logits) or target is None or logits.ndim != 2:
        return None
    labels = torch.as_tensor(target, device=logits.device).reshape(-1).long()
    if labels.numel() != logits.shape[0]:
        return None
    valid = (labels >= 0) & (labels < logits.shape[1])
    if not bool(valid.any()):
        return None
    return F.cross_entropy(logits[valid], labels[valid])


def _serializable(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_serializable(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


class HCFDGTrainer:
    """Train one frozen HCF-DG candidate with the Stage0–4 controller."""

    def __init__(
        self,
        model: Any,
        config: HCFDGConfig | None = None,
        labeled_loader: Iterable[Any] | Callable[[], Iterable[Any]] | None = None,
        unlabeled_loader: Iterable[Any] | Callable[[], Iterable[Any]] | None = None,
        validation_loader: Iterable[Any] | Callable[[], Iterable[Any]] | None = None,
        build_single_view_batch: Callable[..., Any] | None = None,
        *,
        batch_builder: Callable[..., Any] | None = None,
        satellite_augmentor: Any = None,
        loss_fn: Callable[..., Any] | None = None,
        device: str | torch.device | None = None,
        amp: bool | None = None,
        output_dir: str | Path | None = None,
        metrics_jsonl_path: str | Path | None = None,
        metrics_csv_path: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        source_split: Any = "source",
        fold: Any = None,
        seed: int | None = None,
        stage_budget: StageBudget | None = None,
        target_loader: Any = None,
        query_loader: Any = None,
        truth: Any = None,
        **aliases: Any,
    ):
        # Also accept the common planned positional form
        # HCFDGTrainer(model, labeled_loader, unlabeled_loader, validation_loader, config).
        if isinstance(model, HCFDGConfig) and config is not None and not isinstance(config, HCFDGConfig):
            model, config = config, model
        elif (
            not isinstance(config, HCFDGConfig)
            and isinstance(validation_loader, HCFDGConfig)
        ):
            positional_labeled = config
            positional_unlabeled = labeled_loader
            positional_validation = unlabeled_loader
            config = validation_loader
            labeled_loader = positional_labeled
            unlabeled_loader = positional_unlabeled
            validation_loader = positional_validation

        labeled_loader = aliases.pop("source_labeled_loader", labeled_loader)
        unlabeled_loader = aliases.pop("source_unlabeled_loader", unlabeled_loader)
        validation_loader = aliases.pop("source_validation_loader", validation_loader)
        if build_single_view_batch is None:
            build_single_view_batch = aliases.pop("single_view_builder", None)
        if batch_builder is not None:
            if build_single_view_batch is not None and batch_builder is not build_single_view_batch:
                raise ValueError("provide only one single-view batch builder")
            build_single_view_batch = batch_builder
        if aliases:
            names = ", ".join(sorted(aliases))
            raise TypeError(f"unexpected HCFDGTrainer arguments: {names}")

        if target_loader is not None or query_loader is not None or truth is not None:
            raise ValueError("target/query/truth inputs are forbidden for Phase1 HCF-DG training")
        if source_split is not None and any(
            token in str(source_split).lower() for token in ("target", "query", "truth", "phase2")
        ):
            raise ValueError("source_split must identify source-only data")
        if config is None or not isinstance(config, HCFDGConfig):
            raise TypeError("config must be an HCFDGConfig")

        self.model = model
        self.config = config
        self.labeled_loader = labeled_loader
        self.unlabeled_loader = unlabeled_loader
        self.validation_loader = validation_loader
        self.build_single_view_batch = build_single_view_batch
        self.satellite_augmentor = satellite_augmentor
        self._default_loss_fn = None
        if loss_fn is None:
            try:
                from .losses import compose_hcfdg_loss
            except ImportError:
                loss_fn = None
            else:
                loss_fn = compose_hcfdg_loss
                self._default_loss_fn = compose_hcfdg_loss
        self.loss_fn = loss_fn
        self.source_split = source_split
        self.fold = fold
        self.seed = seed
        self.stage_budget = stage_budget or V2_STAGE_BUDGET
        self.output_dir = Path(output_dir) if output_dir is not None else None

        if self.output_dir is not None:
            metrics_jsonl_path = metrics_jsonl_path or self.output_dir / "metrics.jsonl"
            metrics_csv_path = metrics_csv_path or self.output_dir / "metrics.csv"
            checkpoint_path = checkpoint_path or self.output_dir / "checkpoint.pt"
        self.metrics_jsonl_path = Path(metrics_jsonl_path) if metrics_jsonl_path is not None else None
        self.metrics_csv_path = Path(metrics_csv_path) if metrics_csv_path is not None else None
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None

        self.device = self._resolve_device(device)
        if hasattr(self.model, "to"):
            self.model.to(self.device)
        self.amp_enabled = bool(V1_AMP_ENABLED if amp is None else amp)
        self.amp_enabled = self.amp_enabled and self.device.type == "cuda" and torch.cuda.is_available()
        self.scaler = self._make_scaler(self.amp_enabled)

        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(0 if seed is None else int(seed))
        self._labeled_cursor = _LoaderCursor(self.labeled_loader)
        self._unlabeled_cursor = _LoaderCursor(self.unlabeled_loader)
        self._stage0_toggle = 0
        self._last_model_forward_calls: int | None = None

        self._backbone_params, self._new_head_params, self.parameter_group_names = self._split_parameters()
        self.optimizer = torch.optim.AdamW(
            [
                {"params": self._backbone_params, "lr": V1_BACKBONE_LR},
                {"params": self._new_head_params, "lr": V1_HEAD_LR},
            ],
            weight_decay=1e-4,
        )

    @staticmethod
    def _resolve_device(device: str | torch.device | None) -> torch.device:
        if device is not None:
            return torch.device(device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _make_scaler(enabled: bool):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except (AttributeError, TypeError):
            return torch.cuda.amp.GradScaler(enabled=enabled)

    def _named_parameters(self) -> list[tuple[str, nn.Parameter]]:
        if not hasattr(self.model, "named_parameters"):
            return []
        return [(name, parameter) for name, parameter in self.model.named_parameters() if parameter.requires_grad]

    def _split_parameters(self) -> tuple[list[nn.Parameter], list[nn.Parameter], tuple[str, str]]:
        named = self._named_parameters()
        if not named:
            fallback_backbone = nn.Parameter(torch.zeros((), device=self.device))
            fallback_head = nn.Parameter(torch.zeros((), device=self.device))
            self._fallback_parameters = (fallback_backbone, fallback_head)
            return [fallback_backbone], [fallback_head], ("backbone", "new_head")

        backbone_markers = (
            "backbone",
            "identity_backbone",
            "f_shared",
            "shared",
            "sinc",
            "time_domain",
            "temporal",
            "early",
        )
        backbone: list[tuple[str, nn.Parameter]] = []
        head: list[tuple[str, nn.Parameter]] = []
        for name, parameter in named:
            lowered = name.lower()
            if any(marker in lowered for marker in backbone_markers):
                backbone.append((name, parameter))
            else:
                head.append((name, parameter))

        if not backbone:
            backbone.append(head.pop(0))
        if not head:
            head.append(backbone.pop())

        return (
            [parameter for _, parameter in backbone],
            [parameter for _, parameter in head],
            ("backbone", "new_head"),
        )

    @staticmethod
    def _is_v2(config: HCFDGConfig) -> bool:
        candidate_id = str(config.candidate_id).strip().upper()
        return candidate_id in {f"A{i}" for i in range(6, 13)} or int(config.optimizer_updates) == V2_OPTIMIZER_UPDATES

    def _set_learning_rates_and_margin(self, update: int, total_updates: int) -> tuple[float, float]:
        backbone_lr = learning_rate_at(
            update,
            total_updates=total_updates,
            base_lr=V1_BACKBONE_LR,
        )
        head_lr = learning_rate_at(
            update,
            total_updates=total_updates,
            base_lr=V1_HEAD_LR,
        )
        self.optimizer.param_groups[0]["lr"] = backbone_lr
        self.optimizer.param_groups[1]["lr"] = head_lr
        margin = cosface_margin_at(update, total_updates=total_updates)
        for method_name in ("set_cosface_margin", "set_margin"):
            setter = getattr(self.model, method_name, None)
            if callable(setter):
                _invoke_supported(setter, margin)
                break
        else:
            for module_name in ("cosface", "cosface_head", "common_head", "identity_head"):
                module = getattr(self.model, module_name, None)
                if module is not None and hasattr(module, "margin"):
                    module.margin = margin
                    break
        return backbone_lr, margin

    def _next_source_batch(self, *, stage0: bool) -> tuple[Any, float]:
        started = time.perf_counter()
        if stage0 and self.unlabeled_loader is not None:
            # Alternate L_s and U_s without ever requiring U_s to expose TX.
            if self.labeled_loader is not None and self._stage0_toggle % 2:
                raw = self._labeled_cursor.next()
            else:
                raw = self._unlabeled_cursor.next()
            self._stage0_toggle += 1
        else:
            raw = self._labeled_cursor.next()
        return raw, time.perf_counter() - started

    def _prepare_source_batch(self, raw: Any, *, allow_tx: bool, stage0: bool = False) -> _SourceBatch:
        batch = _source_batch(raw, allow_tx=allow_tx, stage0=stage0)
        batch.iq = _move_value(batch.iq, self.device)
        batch.receiver = _move_value(batch.receiver, self.device)
        batch.day = _move_value(batch.day, self.device)
        batch.channel = _move_value(batch.channel, self.device)
        batch.tx = _move_value(batch.tx, self.device)
        batch.domain = _move_value(batch.domain, self.device)
        batch.query_domain = _move_value(batch.query_domain, self.device)
        batch.support_mask = _move_value(batch.support_mask, self.device)
        batch.query_mask = _move_value(batch.query_mask, self.device)
        batch.content_keys = _move_value(batch.content_keys, self.device)
        batch.groups = _move_value(batch.groups, self.device)
        batch.env_meta = _move_value(batch.env_meta, self.device)
        return batch

    def _single_view(self, batch: _SourceBatch) -> Any:
        if self.build_single_view_batch is None:
            return batch.iq
        result = _invoke_supported(
            self.build_single_view_batch,
            batch.iq,
            self.satellite_augmentor,
            self._generator,
            p_sat=0.30,
        )
        channel_labels = _mapping_value(result, ("channel_labels", "channel_label"))
        channel_factors = _mapping_value(result, ("channel_factors", "factors"))
        satellite_mask = _mapping_value(result, ("satellite_mask", "satellite"))
        if channel_labels is not None:
            batch.channel = channel_labels
            batch.env_meta["channel"] = channel_labels
            if batch.env_meta.get("episode_type") == "channel":
                valid = batch.env_meta.get("valid_tx_mask")
                valid = torch.ones_like(channel_labels, dtype=torch.bool) if valid is None else valid.bool()
                query = valid & channel_labels.bool()
                support = valid & ~channel_labels.bool()
                if batch.tx is not None:
                    eligible = torch.zeros_like(valid)
                    for tx_id in torch.unique(batch.tx):
                        tx_mask = valid & batch.tx.eq(tx_id)
                        if bool((query & tx_mask).any()) and bool((support & tx_mask).any()):
                            eligible |= tx_mask
                    query &= eligible
                    support &= eligible
                if not bool(query.any()) or not bool(support.any()):
                    raise ValueError("channel episode requires both clean and satellite support")
                batch.domain = channel_labels
                batch.query_domain = 1
                batch.query_mask = query
                batch.support_mask = support
        if channel_factors is not None:
            batch.env_meta["channel_factors"] = channel_factors
            batch.env_meta["q_phys"] = channel_factors
        if satellite_mask is not None:
            batch.env_meta["satellite_mask"] = satellite_mask
        return _mapping_value(result, ("iq", "x", "inputs", "signal", "IQ")) if isinstance(result, Mapping) else getattr(result, "iq", result)

    def _call_model(self, batch: _SourceBatch, *, update: int, stage: str) -> Any:
        function = self.model.forward if hasattr(self.model, "forward") else self.model
        before = getattr(self.model, "backbone_forward_calls", None)
        result = _invoke_supported(
            function,
            batch.iq,
            tx_labels=batch.tx,
            env_meta=batch.env_meta,
            q_phys=batch.env_meta.get("q_phys"),
            receiver_labels=batch.receiver,
            day_labels=batch.day,
            channel_labels=batch.channel,
            training_aux=True,
            update=update,
            stage=stage,
        )
        after = getattr(self.model, "backbone_forward_calls", None)
        if isinstance(before, int) and isinstance(after, int) and after > before:
            self._last_model_forward_calls = after - before
        else:
            self._last_model_forward_calls = 1
        return result

    def _loss_from_result(
        self,
        result: Any,
        batch: _SourceBatch,
        *,
        update: int,
        stage: str,
        allow_tx: bool,
        include_environment: bool = False,
    ) -> Any:
        if self.loss_fn is not None:
            if self.loss_fn is self._default_loss_fn:
                model_loss = _as_loss(result)
                if torch.is_tensor(model_loss) and model_loss.ndim == 0:
                    return model_loss
            groups = batch.groups
            if groups is None:
                groups = {
                    key: value
                    for key, value in {
                        "receiver": batch.receiver,
                        "day": batch.day,
                        "channel": batch.channel,
                        "tx_receiver": (batch.tx, batch.receiver),
                        "tx_day": (batch.tx, batch.day),
                        "tx_channel": (batch.tx, batch.channel),
                    }.items()
                    if value is not None
                    and not (
                        isinstance(value, tuple)
                        and any(item is None for item in value)
                    )
                }
            result_loss = _invoke_loss_function(
                self.loss_fn,
                result,
                labels=batch.tx if allow_tx else None,
                y=batch.tx if allow_tx else None,
                tx_labels=batch.tx if allow_tx else None,
                env_meta=batch.env_meta,
                domain=batch.domain if batch.domain is not None else batch.receiver,
                receiver=batch.receiver,
                receiver_labels=batch.receiver,
                day=batch.day,
                day_labels=batch.day,
                channel=batch.channel,
                channel_labels=batch.channel,
                content_keys=batch.content_keys,
                query_domain=batch.query_domain,
                support_mask=batch.support_mask,
                query_mask=batch.query_mask,
                groups=groups,
                config=self.config,
                use_lodo=getattr(self.config, "use_lodo", None),
                use_content_conditioning=getattr(self.config, "use_content_conditioning", None),
                use_counterfactual=(
                    str(getattr(self.config, "counterfactual_mode", "off")).lower()
                    not in {"", "off", "none"}
                ),
                use_hdro=getattr(self.config, "use_hdro", None),
                use_csd=getattr(self.config, "use_csd", None),
                use_fac=getattr(self.config, "use_environment_encoder", None),
                update=update,
                stage=stage,
                include_environment=include_environment,
            )
            return _as_loss(result_loss)

        result_loss = _as_loss(result)
        if torch.is_tensor(result_loss) and result_loss.ndim == 0:
            return result_loss

        logits = None
        if isinstance(result, Mapping):
            for key in ("common_logits", "logits"):
                if key in result:
                    logits = result[key]
                    break
        else:
            for key in ("common_logits", "logits"):
                if hasattr(result, key):
                    logits = getattr(result, key)
                    break
        if allow_tx and batch.tx is not None and torch.is_tensor(logits):
            labels = batch.tx.reshape(-1).long()
            if logits.shape[0] == labels.shape[0]:
                return F.cross_entropy(logits, labels)
        if torch.is_tensor(result) and result.numel() == 1:
            return result.reshape(())
        return None

    def _stage0_loss(self, batch: _SourceBatch, *, update: int) -> Any:
        # Stage0 deliberately has no ``tx`` argument and cannot call the
        # identity model.  The planned model may expose any one of these
        # equivalent environment-only hooks while it is landing.
        for method_name in ("stage0_step", "pretrain_environment", "environment_pretrain_step"):
            function = getattr(self.model, method_name, None)
            if callable(function):
                result = _invoke_supported(
                    function,
                    iq=batch.iq,
                    receiver=batch.receiver,
                    day=batch.day,
                    env_meta=batch.env_meta,
                    update=update,
                    stage="stage0",
                )
                return _as_loss(result)
        encoder = getattr(self.model, "environment_encoder", None)
        if callable(encoder) and torch.is_tensor(batch.iq):
            h_early, q_phys = _cheap_environment_inputs(
                batch.iq,
                int(getattr(encoder, "input_dim", 160)),
            )
            result = _invoke_supported(
                encoder,
                h_early,
                q_phys=q_phys,
                env_meta=batch.env_meta,
                receiver_labels=batch.receiver,
                day_labels=batch.day,
            )
            terms = []
            for name, target in (("receiver_logits", batch.receiver), ("day_logits", batch.day)):
                term = _safe_environment_cross_entropy(_object_value(result, name), target)
                if term is not None:
                    terms.append(term)
            if terms:
                return torch.stack(terms).mean()
            scalar = _as_loss(result)
            if torch.is_tensor(scalar) and scalar.ndim == 0:
                return scalar
        return None

    def _extra_environment_loss(self, batch: _SourceBatch, *, update: int, stage: str) -> Any:
        for method_name in ("environment_step", "environment_loss", "auxiliary_environment_step"):
            function = getattr(self.model, method_name, None)
            if callable(function):
                result = _invoke_supported(
                    function,
                    iq=batch.iq,
                    receiver=batch.receiver,
                    day=batch.day,
                    channel=batch.channel,
                    env_meta=batch.env_meta,
                    update=update,
                    stage=stage,
                )
                return _as_loss(result)
        return None

    def _zero_attached_loss(self, loss: Any) -> torch.Tensor:
        if loss is None:
            raise ValueError("training objective must be a finite scalar tensor with gradients")
        if not torch.is_tensor(loss):
            try:
                loss = torch.as_tensor(loss, device=self.device, dtype=torch.float32)
            except (TypeError, ValueError) as exc:
                raise TypeError("training objective must be a finite scalar tensor with gradients") from exc
        if loss.ndim != 0:
            raise ValueError("training objective must be a scalar")
        if not loss.is_floating_point():
            raise TypeError("training objective must be a floating-point scalar")
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("training objective must be finite")
        if not loss.requires_grad:
            raise ValueError("training objective must require gradients")
        return loss.to(self.device) if loss.device != self.device else loss

    def _trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for group in self.optimizer.param_groups for parameter in group["params"] if parameter.requires_grad]

    def _optimizer_step(self, loss: Any) -> tuple[float, float]:
        loss = self._zero_attached_loss(loss)
        self.optimizer.zero_grad(set_to_none=True)
        backward_started = time.perf_counter()
        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        return float(loss.detach().cpu().item()), time.perf_counter() - backward_started

    def _freeze_frontend(self) -> tuple[str, ...]:
        for method_name in (
            "freeze_sinc_and_first_time_domain_block",
            "freeze_sinc_and_first_temporal_block",
            "freeze_frontend_at_stage2",
        ):
            function = getattr(self.model, method_name, None)
            if callable(function):
                result = _invoke_supported(function)
                if result is None:
                    return ()
                if isinstance(result, str):
                    return (result,)
                return tuple(str(item) for item in result)

        frozen: list[str] = []
        named = list(self.model.named_parameters()) if hasattr(self.model, "named_parameters") else []
        names_by_parameter = {id(parameter): name for name, parameter in named}

        def freeze_module(module: Any) -> list[str]:
            if not isinstance(module, nn.Module):
                return []
            selected: list[str] = []
            for parameter in module.parameters():
                parameter.requires_grad_(False)
                name = names_by_parameter.get(id(parameter))
                if name is not None and name not in frozen:
                    selected.append(name)
            return selected

        roots = [self.model]
        backbone = getattr(self.model, "backbone", None)
        if isinstance(backbone, nn.Module) and backbone is not self.model:
            roots.append(backbone)

        for root in roots:
            sinc_module = next(
                (
                    getattr(root, attribute, None)
                    for attribute in ("sinc", "sinc_layer", "sinc_conv", "sincnet")
                    if isinstance(getattr(root, attribute, None), nn.Module)
                ),
                None,
            )
            if sinc_module is not None:
                frozen.extend(freeze_module(sinc_module))
                break

        for root in roots:
            time_module = next(
                (
                    getattr(root, attribute, None)
                    for attribute in (
                        "first_time_domain_block",
                        "time_domain_block",
                        "t1",
                        "temporal_block",
                        "temporal_blocks",
                        "time_fuse",
                    )
                    if isinstance(getattr(root, attribute, None), nn.Module)
                ),
                None,
            )
            if isinstance(time_module, (nn.ModuleList, nn.Sequential)):
                time_module = time_module[0] if len(time_module) else None
            if time_module is not None:
                frozen.extend(freeze_module(time_module))
                break

        for name, parameter in named:
            if "sinc" in name.lower() and name not in frozen:
                parameter.requires_grad_(False)
                frozen.append(name)

        temporal_tokens = {"time_domain", "temporal", "time_conv", "td_block", "t1"}
        temporal_candidates = [
            (name, parameter)
            for name, parameter in named
            if any(part.lower() in temporal_tokens for part in name.split("."))
        ]
        if temporal_candidates:
            candidate_name = temporal_candidates[0][0]
            parts = candidate_name.split(".")
            token_position = next(
                (position for position, part in enumerate(parts) if part.lower() in temporal_tokens),
                max(0, len(parts) - 2),
            )
            first_prefix = ".".join(parts[: token_position + 1])
            for name, parameter in named:
                if name == first_prefix or name.startswith(first_prefix + "."):
                    parameter.requires_grad_(False)
                    if name not in frozen:
                        frozen.append(name)
        return tuple(frozen)

    def _peak_memory(self) -> float:
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return 0.0
        return float(torch.cuda.max_memory_allocated(self.device))

    def _total_gpu_hours(self, elapsed_seconds: float) -> float:
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return 0.0
        return elapsed_seconds * max(1, torch.cuda.device_count()) / 3600.0

    def _record_metrics(
        self,
        state: TrainState,
        *,
        update: int,
        stage: str,
        samples: int,
        step_time: float,
        dataloader_wait: float,
        forward_time: float,
        backward_time: float,
        loss: float,
        backbone_lr: float,
        margin: float,
        elapsed_seconds: float,
    ) -> None:
        samples_per_second = samples / step_time if step_time > 0.0 else 0.0
        total_gpu_hours = self._total_gpu_hours(elapsed_seconds)
        row = {
            "update": update,
            "optimizer_update": update,
            "stage": stage,
            "loss": loss,
            "step_time": step_time,
            "samples": samples,
            "samples/s": samples_per_second,
            "samples_per_second": samples_per_second,
            "dataloader_wait": dataloader_wait,
            "peak_memory": self._peak_memory(),
            "peak_memory_bytes": self._peak_memory(),
            "forward_time": forward_time,
            "backward_time": backward_time,
            "backbone_lr": backbone_lr,
            "head_lr": float(self.optimizer.param_groups[1]["lr"]),
            "cosface_margin": margin,
            "amp_enabled": self.amp_enabled,
            "backbone_forward_calls": state.backbone_forward_calls,
            "total_gpu_hours": total_gpu_hours,
            "gpu_hours": total_gpu_hours,
        }
        state.metrics.append(row)

    def _write_metrics(self, metrics: list[dict[str, Any]]) -> None:
        if not metrics:
            return
        if self.metrics_jsonl_path is not None:
            self.metrics_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with self.metrics_jsonl_path.open("w", encoding="utf-8", newline="\n") as handle:
                for row in metrics:
                    handle.write(json.dumps(_serializable(row), ensure_ascii=False, sort_keys=True) + "\n")
        if self.metrics_csv_path is not None:
            self.metrics_csv_path.parent.mkdir(parents=True, exist_ok=True)
            fieldnames: list[str] = []
            for row in metrics:
                for key in row:
                    if key not in fieldnames:
                        fieldnames.append(key)
            with self.metrics_csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for row in metrics:
                    writer.writerow(_serializable(row))

    def checkpoint_payload(self, state: TrainState) -> CheckpointPayload:
        model_state = self.model.state_dict() if hasattr(self.model, "state_dict") else {}
        config_value = asdict(self.config) if is_dataclass(self.config) else dict(vars(self.config))
        inference = {
            "head": "common",
            "common_head_only": True,
            "uses_environment_encoder": False,
            "uses_specific_head": False,
            "uses_counterfactual_transport": False,
            "phase1_inference_graph": "IQ -> F_shared -> P_id -> z_id -> W0",
        }
        return CheckpointPayload(
            phase1_method="hcfdg",
            candidate_id=str(self.config.candidate_id),
            source_split=self.source_split,
            fold=self.fold,
            seed=self.seed,
            update=state.optimizer_updates,
            config=config_value,
            model_state=model_state,
            optimizer_state=self.optimizer.state_dict(),
            scaler_state=self.scaler.state_dict(),
            inference=inference,
        )

    def save_checkpoint(self, payload: CheckpointPayload | None = None) -> Path | None:
        if self.checkpoint_path is None:
            return None
        payload = payload or self.checkpoint_payload(TrainState(optimizer_updates=int(self.config.optimizer_updates)))
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload.to_dict(), self.checkpoint_path)
        return self.checkpoint_path

    def train(self, config: HCFDGConfig | None = None) -> TrainState:
        """Run the candidate's exact update budget and return its final state."""

        if config is not None:
            if not isinstance(config, HCFDGConfig):
                raise TypeError("config must be an HCFDGConfig")
            self.config = config
        total_updates = int(self.config.optimizer_updates)
        if total_updates <= 0:
            raise ValueError("optimizer_updates must be positive")
        is_v2 = self._is_v2(self.config)
        if is_v2:
            expected = self.stage_budget.total_updates
            if total_updates != expected:
                raise ValueError(f"V2 HCF-DG requires exactly {expected} optimizer updates")

        if hasattr(self.model, "train"):
            self.model.train()
        state = TrainState(amp_enabled=self.amp_enabled)
        if is_v2:
            state.stage_updates = {"stage0": 0, "stage1": 0, "stage2": 0, "stage3": 0, "stage4": 0}
        else:
            state.stage_updates = {"v1": 0}

        started = time.perf_counter()
        for update in range(1, total_updates + 1):
            step_started = time.perf_counter()
            backbone_lr, margin = self._set_learning_rates_and_margin(update, total_updates)
            stage = stage_for_update(update, self.stage_budget) if is_v2 else "v1"
            is_stage0 = stage == "stage0"
            raw, dataloader_wait = self._next_source_batch(stage0=is_stage0)
            source_batch = self._prepare_source_batch(raw, allow_tx=not is_stage0, stage0=is_stage0)
            samples = int(source_batch.iq.shape[0]) if torch.is_tensor(source_batch.iq) and source_batch.iq.ndim > 0 else 0

            forward_started = time.perf_counter()
            if is_stage0:
                result_loss = self._stage0_loss(source_batch, update=update)
            else:
                source_batch.iq = self._single_view(source_batch)
                result = self._call_model(source_batch, update=update, stage=stage)
                result_loss = self._loss_from_result(
                    result,
                    source_batch,
                    update=update,
                    stage=stage,
                    allow_tx=True,
                )
            forward_time = time.perf_counter() - forward_started

            main_update = update - self.stage_budget.stage0 if is_v2 else update
            if is_v2 and not is_stage0 and main_update % self.stage_budget.environment_update_interval == 0:
                extra_raw, extra_wait = self._next_source_batch(stage0=True)
                dataloader_wait += extra_wait
                extra_batch = self._prepare_source_batch(extra_raw, allow_tx=False)
                extra_loss = self._extra_environment_loss(extra_batch, update=update, stage=stage)
                if extra_loss is not None:
                    result_loss = self._zero_attached_loss(result_loss) + self._zero_attached_loss(extra_loss)
                state.environment_updates += 1
            if is_stage0:
                state.environment_updates += 1

            loss_value, backward_time = self._optimizer_step(result_loss)
            state.optimizer_updates += 1
            state.stage_updates[stage] += 1
            if self._last_model_forward_calls is not None:
                state.backbone_forward_calls += self._last_model_forward_calls
                self._last_model_forward_calls = None

            if is_v2 and update == int(total_updates * self.stage_budget.freeze_progress):
                self._freeze_frontend()
                state.freeze_update = update

            elapsed = time.perf_counter() - started
            self._record_metrics(
                state,
                update=update,
                stage=stage,
                samples=samples,
                step_time=time.perf_counter() - step_started,
                dataloader_wait=dataloader_wait,
                forward_time=forward_time,
                backward_time=backward_time,
                loss=loss_value,
                backbone_lr=backbone_lr,
                margin=margin,
                elapsed_seconds=elapsed,
            )

        state.elapsed_seconds = time.perf_counter() - started
        state.total_gpu_hours = self._total_gpu_hours(state.elapsed_seconds)
        self._write_metrics(state.metrics)
        state.checkpoint = self.checkpoint_payload(state)
        if self.checkpoint_path is not None:
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(state.checkpoint.to_dict(), self.checkpoint_path)
        return state


__all__ = [
    "CheckpointPayload",
    "HCFDGTrainer",
    "TrainState",
    "cosface_margin_at",
    "learning_rate_at",
    "stage_for_update",
    "warmup_updates",
]
