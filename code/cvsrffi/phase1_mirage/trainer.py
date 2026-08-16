"""Source-only EMA/SWAD fold training with immutable MIRAGE run receipts.

This module intentionally owns neither calibration nor target scoring.  It
accepts only the four approved source roles, uses ``V_select`` for model
selection, and records ``V_cal`` as diagnostic input for the later calibration
stage.
"""

from __future__ import annotations

import copy
import csv
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import random
import re
from types import MappingProxyType
from typing import Iterable, Literal, Mapping, Sequence
import uuid

import numpy as np
import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from .config import ArmConfig, arm_config
from .head import MIRAGEOpenHead, OpenHeadOutput
from .losses import build_boundary_mixup, compute_arm_losses
from .model import MIRAGEEncoder
from .proxy import ProxyEpisode, build_proxy_episode


_REQUIRED_LOADER_ROLES = frozenset({"l", "u", "v_cal", "v_select"})
_TARGET_LOADER_ROLES = frozenset({"target", "target_known", "target_unknown"})
_RUN_ARTIFACTS = frozenset(
    {
        "checkpoint.pt",
        "metrics_epoch.jsonl",
        "metrics_epoch.csv",
        "split_receipt.json",
        "proxy_receipt.json",
        "resource_receipt.json",
        "completion_receipt.json",
    }
)
_FORMAL_EPOCHS = 200
_WARMUP_END = 40
_JOINT_END = 160
_EMA_DECAY = 0.999
_GIT_COMMIT_PATTERN = re.compile(r"^[0-9a-fA-F]{7,64}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


class TrainingProtocolError(ValueError):
    """Raised for a source-role, lifecycle, or immutable-artifact violation."""


@dataclass(frozen=True)
class TrainConfig:
    """Frozen controls for one source-only causal-arm fold.

    ``git_commit`` and ``split_sha256`` are explicit read-only provenance
    inputs.  The trainer never runs a shell command to discover either value.
    """

    arm: Literal["B0", "A", "B", "C"]
    epochs: int = _FORMAL_EPOCHS
    warmup_end: int = _WARMUP_END
    joint_end: int = _JOINT_END
    formal: bool = True
    num_classes: int = 3
    covariance_rank: int = 8
    device: str = "cpu"
    ema_decay: float = _EMA_DECAY
    proxy_seed: int = 0
    git_commit: str = "UNSPECIFIED"
    split_sha256: str = ""

    def __post_init__(self) -> None:
        if self.arm not in {"B0", "A", "B", "C"}:
            raise TrainingProtocolError("arm must be one of B0, A, B, C")
        # Resolving the frozen config here makes an invalid arm fail before any
        # output directory or compute resource is touched.
        frozen = arm_config(self.arm)
        if not isinstance(self.formal, bool):
            raise TrainingProtocolError("formal must be a bool")
        for field_name in ("epochs", "warmup_end", "joint_end", "num_classes", "covariance_rank", "proxy_seed"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TrainingProtocolError(f"{field_name} must be an integer")
        if self.epochs < 1:
            raise TrainingProtocolError("epochs must be positive")
        if self.warmup_end != _WARMUP_END or self.joint_end != _JOINT_END:
            raise TrainingProtocolError("MIRAGE stages must remain 1-40, 41-160, 161-200")
        if self.formal and self.epochs != frozen.epochs:
            raise TrainingProtocolError("formal MIRAGE training requires 200 epochs")
        if self.num_classes < 3:
            raise TrainingProtocolError("num_classes must be at least three for the frozen proxy schedule")
        if self.covariance_rank < 0:
            raise TrainingProtocolError("covariance_rank must be non-negative")
        if not isinstance(self.device, str) or not self.device:
            raise TrainingProtocolError("device must be a non-empty string")
        if not isinstance(self.ema_decay, float) or self.ema_decay != _EMA_DECAY:
            raise TrainingProtocolError("ema_decay must equal frozen value 0.999")
        if not isinstance(self.git_commit, str):
            raise TrainingProtocolError("git_commit must be an explicit string")
        if not isinstance(self.split_sha256, str):
            raise TrainingProtocolError("split_sha256 must be an explicit string")


@dataclass(frozen=True)
class TrainResult:
    """Paths and immutable completion facts for one closed source fold."""

    checkpoint_path: Path
    completion_receipt: Mapping[str, object]


@dataclass
class EMAState:
    """Detached teacher copies updated only after an optimizer step."""

    model: MIRAGEEncoder
    head: MIRAGEOpenHead
    updates: int = 0


@dataclass
class SWADAccumulator:
    """CPU sums of EMA weights from the formal stabilization interval only."""

    model_sums: dict[str, Tensor]
    head_sums: dict[str, Tensor]
    count: int = 0


@dataclass(frozen=True)
class _LabeledBatch:
    iq: Tensor
    labels: Tensor
    receiver_ids: Tensor
    day_ids: Tensor
    scene_ids: Tensor


@dataclass(frozen=True)
class _UnlabeledBatch:
    weak_iq: Tensor
    strong_iq: Tensor


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clone_state_dict(module: nn.Module) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu().clone().contiguous()
        for name, value in module.state_dict().items()
    }


def _state_dict_sha256(state: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, Tensor):
            raise TrainingProtocolError(f"state entry {name} must be a tensor")
        canonical = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(canonical.dtype).encode("ascii"))
        digest.update(_canonical_json(tuple(canonical.shape)).encode("ascii"))
        # Scalar parameters such as ``risk_bias`` need a one-dimensional view
        # before reinterpreting their storage as bytes.
        digest.update(canonical.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _nested_state_sha256(*states: Mapping[str, Tensor]) -> str:
    return _canonical_sha256([_state_dict_sha256(state) for state in states])


def _capture_rng_state() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().cpu(),
        "cuda": tuple(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else None,
    }


def _validate_explicit_metadata(config: TrainConfig) -> None:
    if not _GIT_COMMIT_PATTERN.fullmatch(config.git_commit):
        raise TrainingProtocolError("git_commit must be an explicit 7-64 character hexadecimal commit")
    if not _SHA256_PATTERN.fullmatch(config.split_sha256):
        raise TrainingProtocolError("split_sha256 must be an explicit SHA256 digest")


def _validate_loader_schema(loaders: Mapping[str, object]) -> None:
    if not isinstance(loaders, Mapping):
        raise TrainingProtocolError("loaders must be a mapping")
    keys = set(loaders)
    forbidden = sorted(_TARGET_LOADER_ROLES.intersection(keys))
    if forbidden:
        raise TrainingProtocolError(f"target loader is forbidden: {', '.join(forbidden)}")
    if keys != _REQUIRED_LOADER_ROLES:
        missing = sorted(_REQUIRED_LOADER_ROLES - keys)
        extra = sorted(keys - _REQUIRED_LOADER_ROLES)
        raise TrainingProtocolError(
            "loader schema must equal {l, u, v_cal, v_select}; "
            f"missing={missing}; extra={extra}"
        )
    for role, loader in loaders.items():
        if not isinstance(role, str) or not isinstance(loader, Iterable):
            raise TrainingProtocolError(f"loader {role!r} must be an iterable source loader")


def _stage_for_epoch(epoch: int, config: TrainConfig) -> str:
    if isinstance(epoch, bool) or not isinstance(epoch, int) or not 1 <= epoch <= config.epochs:
        raise TrainingProtocolError("epoch must be within the configured training interval")
    if epoch <= config.warmup_end:
        return "warmup"
    if epoch <= config.joint_end:
        return "joint"
    return "stabilization"


def _resolve_device(config: TrainConfig) -> torch.device:
    try:
        device = torch.device(config.device)
    except (TypeError, RuntimeError) as error:
        raise TrainingProtocolError(f"invalid device: {config.device}") from error
    if device.type == "cuda" and not torch.cuda.is_available():
        raise TrainingProtocolError("CUDA device requested but CUDA is unavailable")
    return device


def build_models_and_optimizer(config: TrainConfig) -> tuple[MIRAGEEncoder, MIRAGEOpenHead, torch.optim.Optimizer]:
    """Build the frozen-capacity model, head, and optimizer for one arm."""

    if not isinstance(config, TrainConfig):
        raise TypeError("config must be a TrainConfig")
    frozen: ArmConfig = arm_config(config.arm)
    device = _resolve_device(config)
    model = MIRAGEEncoder(frozen.encoder).to(device)
    head = MIRAGEOpenHead(
        num_classes=config.num_classes,
        feature_dim=frozen.encoder.z_id_dim,
        covariance_rank=config.covariance_rank,
    ).to(device)
    optimizer = torch.optim.AdamW(
        tuple(model.parameters()) + tuple(head.parameters()),
        lr=frozen.learning_rate,
        weight_decay=frozen.weight_decay,
    )
    return model, head, optimizer


def make_ema_copy(model: MIRAGEEncoder, head: MIRAGEOpenHead) -> EMAState:
    """Create an exact detached source teacher before the first optimizer step."""

    ema_model = copy.deepcopy(model).eval()
    ema_head = copy.deepcopy(head).eval()
    for parameter in tuple(ema_model.parameters()) + tuple(ema_head.parameters()):
        parameter.requires_grad_(False)
    return EMAState(model=ema_model, head=ema_head)


def make_swad_accumulator(model: MIRAGEEncoder, head: MIRAGEOpenHead) -> SWADAccumulator:
    """Allocate empty CPU accumulators; only ``update_swad`` may add EMA states."""

    return SWADAccumulator(
        model_sums={name: torch.zeros_like(value.detach().cpu()) for name, value in model.state_dict().items()},
        head_sums={name: torch.zeros_like(value.detach().cpu()) for name, value in head.state_dict().items()},
    )


def update_ema(ema: EMAState, model: MIRAGEEncoder, head: MIRAGEOpenHead, *, decay: float = _EMA_DECAY) -> None:
    """Apply one post-step EMA update without tracking gradients."""

    if not isinstance(ema, EMAState):
        raise TypeError("ema must be an EMAState")
    if decay != _EMA_DECAY:
        raise TrainingProtocolError("EMA decay must equal frozen value 0.999")
    with torch.no_grad():
        for teacher, student in ((ema.model, model), (ema.head, head)):
            for teacher_parameter, student_parameter in zip(teacher.parameters(), student.parameters(), strict=True):
                teacher_parameter.mul_(decay).add_(student_parameter.detach(), alpha=1.0 - decay)
            for teacher_buffer, student_buffer in zip(teacher.buffers(), student.buffers(), strict=True):
                teacher_buffer.copy_(student_buffer.detach())
    ema.updates += 1


def update_swad(swad: SWADAccumulator, ema: EMAState) -> None:
    """Add exactly one EMA state to the stabilization-only SWAD average."""

    if not isinstance(swad, SWADAccumulator) or not isinstance(ema, EMAState):
        raise TypeError("swad and ema must use their MIRAGE state types")
    for sums, module in ((swad.model_sums, ema.model), (swad.head_sums, ema.head)):
        state = module.state_dict()
        if set(sums) != set(state):
            raise TrainingProtocolError("SWAD state schema changed during training")
        for name, value in state.items():
            sums[name].add_(value.detach().cpu())
    swad.count += 1


def _swad_state(swad: SWADAccumulator, ema: EMAState) -> tuple[dict[str, Tensor], dict[str, Tensor], str]:
    if swad.count == 0:
        return (
            _clone_state_dict(ema.model),
            _clone_state_dict(ema.head),
            "EMA_FALLBACK_NO_COMPLETE_161_200_WINDOW",
        )
    return (
        {name: value.div(float(swad.count)).clone() for name, value in swad.model_sums.items()},
        {name: value.div(float(swad.count)).clone() for name, value in swad.head_sums.items()},
        "SWAD_EMA_E161_E200",
    )


def _mapping_batch(batch: object, *, role: str) -> Mapping[str, object]:
    if not isinstance(batch, Mapping):
        raise TrainingProtocolError(f"{role} batch must be a mapping")
    return batch


def _float_tensor(value: object, *, name: str, device: torch.device) -> Tensor:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise TrainingProtocolError(f"{name} must be a floating torch.Tensor")
    return value.to(device=device, dtype=torch.float32, non_blocking=False)


def _integer_tensor(value: object, *, name: str, device: torch.device, length: int | None = None) -> Tensor:
    if not isinstance(value, Tensor) or value.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise TrainingProtocolError(f"{name} must be an integer torch.Tensor")
    result = value.to(device=device, dtype=torch.int64, non_blocking=False)
    if result.ndim != 1 or (length is not None and result.numel() != length):
        raise TrainingProtocolError(f"{name} must have shape [B]")
    return result


def _labeled_batch(batch: object, *, role: str, device: torch.device, require_groups: bool) -> _LabeledBatch:
    values = _mapping_batch(batch, role=role)
    allowed = {"iq", "labels", "receiver_ids", "day_ids", "scene_ids"}
    unexpected = sorted(set(values) - allowed)
    if unexpected:
        raise TrainingProtocolError(f"{role} batch has unsupported fields: {unexpected}")
    if not {"iq", "labels"}.issubset(values):
        raise TrainingProtocolError(f"{role} batch requires iq and labels")
    iq = _float_tensor(values["iq"], name=f"{role}.iq", device=device)
    if iq.ndim != 3 or iq.shape[1] != 2:
        raise TrainingProtocolError(f"{role}.iq must have shape [B, 2, T]")
    labels = _integer_tensor(values["labels"], name=f"{role}.labels", device=device, length=iq.shape[0])
    if bool((labels < 0).any()):
        raise TrainingProtocolError(f"{role}.labels must be non-negative")
    group_values: dict[str, Tensor] = {}
    for field_name in ("receiver_ids", "day_ids", "scene_ids"):
        if field_name not in values:
            if require_groups:
                raise TrainingProtocolError(f"{role} requires {field_name} for formal C Group-CVaR")
            group_values[field_name] = torch.zeros(iq.shape[0], dtype=torch.int64, device=device)
        else:
            group_values[field_name] = _integer_tensor(
                values[field_name], name=f"{role}.{field_name}", device=device, length=iq.shape[0]
            )
    return _LabeledBatch(iq=iq, labels=labels, **group_values)


def _unlabeled_batch(batch: object, *, device: torch.device) -> _UnlabeledBatch:
    values = _mapping_batch(batch, role="u")
    if set(values) != {"weak_iq", "strong_iq"}:
        raise TrainingProtocolError("u batch schema must equal {weak_iq, strong_iq} without labels")
    weak = _float_tensor(values["weak_iq"], name="u.weak_iq", device=device)
    strong = _float_tensor(values["strong_iq"], name="u.strong_iq", device=device)
    if weak.shape != strong.shape or weak.ndim != 3 or weak.shape[1] != 2:
        raise TrainingProtocolError("u weak_iq and strong_iq must share shape [B, 2, T]")
    return _UnlabeledBatch(weak_iq=weak, strong_iq=strong)


def _next_unlabeled(iterator: object, loader: Iterable[object]) -> tuple[object, object]:
    try:
        return next(iterator), iterator
    except StopIteration:
        replacement = iter(loader)
        try:
            return next(replacement), replacement
        except StopIteration as error:
            raise TrainingProtocolError("u loader must not be empty") from error


def _teacher_radius_evidence(output: OpenHeadOutput) -> tuple[Tensor, Tensor]:
    labels = output.class_scores.argmax(dim=1)
    margins = output.radius_margins.gather(1, labels[:, None]).squeeze(1)
    return margins <= 0.0, margins


def _masked_latent_pair(model: MIRAGEEncoder, iq: Tensor, target_tokens: Tensor) -> tuple[Tensor, Tensor]:
    masked_iq = iq.clone()
    masked_iq[..., ::4] = 0.0
    prediction = model(masked_iq).tokens
    if prediction.shape != target_tokens.shape:
        raise TrainingProtocolError("masked latent token shape changed")
    return prediction, target_tokens.detach()


def _proxy_inputs(
    *,
    head: MIRAGEOpenHead,
    embeddings: Tensor,
    labels: Tensor,
    seed: int,
    episode_index: int,
) -> tuple[ProxyEpisode, OpenHeadOutput, object, OpenHeadOutput | None]:
    episode = build_proxy_episode(labels, split_role="train_l", seed=seed, episode_index=episode_index)
    open_output = head(embeddings)
    registered_rows = torch.zeros(labels.shape[0], dtype=torch.bool, device=labels.device)
    registered_rows[episode.registered_rows] = True
    mixup = build_boundary_mixup(embeddings, labels, registered_rows)
    mixup_output = (
        head(mixup.mixed_embeddings, class_mask=episode.registered_class_mask)
        if mixup.mixed_embeddings.shape[0] > 0
        else None
    )
    return episode, open_output, mixup, mixup_output


def run_train_epoch(
    *,
    model: MIRAGEEncoder,
    head: MIRAGEOpenHead,
    ema: EMAState,
    optimizer: torch.optim.Optimizer,
    labeled_loader: Iterable[object],
    unlabeled_loader: Iterable[object],
    epoch: int,
    config: TrainConfig,
) -> dict[str, object]:
    """Run one source-only epoch and update EMA immediately after each step."""

    device = _resolve_device(config)
    stage = _stage_for_epoch(epoch, config)
    model.train()
    head.train()
    ema.model.eval()
    ema.head.eval()
    unlabeled_iterator = iter(unlabeled_loader)
    totals: dict[str, float] = {}
    optimizer_steps = 0
    proxy_schedule: list[dict[str, int | str]] = []
    for batch_index, raw_labeled in enumerate(labeled_loader):
        labeled = _labeled_batch(
            raw_labeled,
            role="l",
            device=device,
            require_groups=config.arm == "C",
        )
        if bool((labeled.labels >= config.num_classes).any()):
            raise TrainingProtocolError("l.labels exceed configured num_classes")
        raw_unlabeled, unlabeled_iterator = _next_unlabeled(unlabeled_iterator, unlabeled_loader)
        unlabeled = _unlabeled_batch(raw_unlabeled, device=device)
        features = model(labeled.iq)
        supervised_output = head(features.z_id)
        supervised_logits = supervised_output.class_scores
        if stage == "warmup":
            losses: dict[str, Tensor] = {
                "registered_ce": functional.cross_entropy(supervised_logits, labeled.labels),
            }
            losses["total"] = losses["registered_ce"]
        else:
            student_output = head(model(unlabeled.strong_iq).z_id)
            with torch.no_grad():
                teacher_output = ema.head(ema.model(unlabeled.weak_iq).z_id)
            pseudo_inside_radius, pseudo_margins = _teacher_radius_evidence(teacher_output)
            loss_inputs: dict[str, object] = {
                "supervised_logits": supervised_logits,
                "supervised_labels": labeled.labels,
                "pseudo_student_logits": student_output.class_scores,
                "pseudo_teacher_logits": teacher_output.class_scores,
                "pseudo_inside_radius": pseudo_inside_radius,
            }
            if config.arm in {"A", "B", "C"}:
                masked_prediction, masked_target = _masked_latent_pair(model, labeled.iq, features.tokens)
                loss_inputs.update(
                    {
                        "pseudo_radius_margins": pseudo_margins,
                        "masked_prediction": masked_prediction,
                        "masked_target": masked_target,
                        "cross_receiver_features": features.z_dom,
                        "cross_receiver_ids": labeled.receiver_ids,
                    }
                )
            if config.arm in {"B", "C"}:
                episode, open_output, mixup, mixup_output = _proxy_inputs(
                    head=head,
                    embeddings=features.z_id,
                    labels=labeled.labels,
                    seed=config.proxy_seed,
                    episode_index=(epoch - 1) * 1_000_000 + batch_index,
                )
                loss_inputs.update(
                    {
                        "proxy_episode": episode,
                        "open_output": open_output,
                        "boundary_embeddings": features.z_id,
                        "boundary_mixup_batch": mixup,
                        "boundary_mixup_output": mixup_output,
                    }
                )
                proxy_schedule.append(dict(episode.schedule_receipt))
            if config.arm == "C":
                loss_inputs.update(
                    {
                        "group_losses": functional.cross_entropy(
                            supervised_logits, labeled.labels, reduction="none"
                        ),
                        "receiver_ids": labeled.receiver_ids,
                        "day_ids": labeled.day_ids,
                        "scene_ids": labeled.scene_ids,
                    }
                )
            losses = compute_arm_losses(config.arm, **loss_inputs)
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        optimizer.step()
        # This call is intentionally adjacent to optimizer.step(): no EMA
        # update occurs before a successful optimizer update.
        update_ema(ema, model, head, decay=config.ema_decay)
        optimizer_steps += 1
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach().cpu())
    if optimizer_steps == 0:
        raise TrainingProtocolError("l loader must not be empty")
    return {
        "stage": stage,
        "optimizer_steps": optimizer_steps,
        "ema_updates": optimizer_steps,
        "ema_updated_after_optimizer_step": True,
        "losses": {name: value / optimizer_steps for name, value in sorted(totals.items())},
        "proxy_schedule": proxy_schedule,
    }


def _known_metrics(model: MIRAGEEncoder, head: MIRAGEOpenHead, loader: Iterable[object], *, role: str, device: torch.device, num_classes: int) -> dict[str, float]:
    class_correct = torch.zeros(num_classes, dtype=torch.float64)
    class_total = torch.zeros(num_classes, dtype=torch.float64)
    scene_correct: dict[int, list[float]] = {}
    batch_count = 0
    for raw_batch in loader:
        batch = _labeled_batch(raw_batch, role=role, device=device, require_groups=False)
        if bool((batch.labels >= num_classes).any()):
            raise TrainingProtocolError(f"{role}.labels exceed configured num_classes")
        output = head(model(batch.iq).z_id)
        predictions = output.class_scores.argmax(dim=1)
        correct = predictions.eq(batch.labels)
        for class_id in range(num_classes):
            class_mask = batch.labels.eq(class_id)
            class_total[class_id] += int(class_mask.sum())
            class_correct[class_id] += int((correct & class_mask).sum())
        for scene_id in batch.scene_ids.unique(sorted=True).detach().cpu().tolist():
            scene_mask = batch.scene_ids.eq(int(scene_id))
            labels = batch.labels[scene_mask]
            scene_values = scene_correct.setdefault(int(scene_id), [0.0] * (2 * num_classes))
            for class_id in range(num_classes):
                class_mask = labels.eq(class_id)
                scene_values[class_id] += float((correct[scene_mask] & class_mask).sum())
                scene_values[num_classes + class_id] += float(class_mask.sum())
        batch_count += 1
    if batch_count == 0:
        raise TrainingProtocolError(f"{role} loader must not be empty")
    observed = class_total > 0
    if not bool(observed.any()):
        raise TrainingProtocolError(f"{role} has no observed known classes")
    macro = float((class_correct[observed] / class_total[observed]).mean())
    scene_macros: list[float] = []
    for values in scene_correct.values():
        correct_values = torch.tensor(values[:num_classes], dtype=torch.float64)
        total_values = torch.tensor(values[num_classes:], dtype=torch.float64)
        present = total_values > 0
        scene_macros.append(float((correct_values[present] / total_values[present]).mean()))
    if not scene_macros:
        raise TrainingProtocolError(f"{role} has no scene statistics")
    return {"known_macro": macro, "worst_scene": min(scene_macros)}


def run_source_validation(
    *,
    model: MIRAGEEncoder,
    head: MIRAGEOpenHead,
    ema: EMAState,
    v_cal: Iterable[object],
    v_select: Iterable[object],
    device: torch.device,
) -> Mapping[str, float]:
    """Evaluate source known metrics without modifying model, EMA, head, or prototypes."""

    modules = (model, head, ema.model, ema.head)
    before_hashes = tuple(_state_dict_sha256(_clone_state_dict(module)) for module in modules)
    modes = tuple(module.training for module in modules)
    try:
        for module in modules:
            module.eval()
        with torch.no_grad():
            calibration = _known_metrics(ema.model, ema.head, v_cal, role="v_cal", device=device, num_classes=ema.head.num_classes)
            selection = _known_metrics(ema.model, ema.head, v_select, role="v_select", device=device, num_classes=ema.head.num_classes)
    finally:
        for module, mode in zip(modules, modes, strict=True):
            module.train(mode)
    after_hashes = tuple(_state_dict_sha256(_clone_state_dict(module)) for module in modules)
    if before_hashes != after_hashes:
        raise TrainingProtocolError("source validation mutated model, EMA, head, or prototype state")
    return MappingProxyType(
        {
            "v_cal_known_macro": calibration["known_macro"],
            "v_cal_worst_scene": calibration["worst_scene"],
            "v_select_known_macro": selection["known_macro"],
            "v_select_worst_scene": selection["worst_scene"],
        }
    )


def _select_best_epoch(history: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    """Select only by the predeclared V_select macro/worst-scene/early rule."""

    if not history:
        raise TrainingProtocolError("cannot select a checkpoint without epoch metrics")
    required = {"epoch", "v_select_known_macro", "v_select_worst_scene"}
    for row in history:
        if not required.issubset(row):
            raise TrainingProtocolError("epoch metrics lack V_select selection fields")
    best = min(
        history,
        key=lambda row: (
            -float(row["v_select_known_macro"]),
            -float(row["v_select_worst_scene"]),
            int(row["epoch"]),
        ),
    )
    selected = dict(best)
    selected["selection_source"] = "V_select"
    return MappingProxyType(selected)


def _atomic_write_bytes(path: Path, payload: bytes, *, allow_replace: bool = False) -> None:
    if path.exists() and not allow_replace:
        raise TrainingProtocolError(f"refusing to overwrite artifact: {path.name}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists() and not allow_replace:
            raise TrainingProtocolError(f"refusing to overwrite artifact: {path.name}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, payload: Mapping[str, object], *, allow_replace: bool = False) -> None:
    _atomic_write_bytes(
        path,
        (json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8"),
        allow_replace=allow_replace,
    )


def _atomic_write_torch(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists():
        raise TrainingProtocolError(f"refusing to overwrite artifact: {path.name}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise TrainingProtocolError(f"refusing to overwrite artifact: {path.name}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def append_epoch_metrics(path: Path, history: Sequence[Mapping[str, object]]) -> None:
    """Atomically replace only the new run's complete JSONL and CSV metrics views."""

    json_lines = "".join(_canonical_json(dict(row)) + "\n" for row in history)
    _atomic_write_bytes(path, json_lines.encode("utf-8"), allow_replace=path.exists())
    csv_path = path.with_name("metrics_epoch.csv")
    flattened: list[dict[str, object]] = []
    for row in history:
        flattened_row = {key: value for key, value in row.items() if key not in {"losses", "proxy_schedule"}}
        for loss_name, loss_value in dict(row["losses"]).items():
            flattened_row[f"loss_{loss_name}"] = loss_value
        flattened.append(flattened_row)
    fields = sorted({field for row in flattened for field in row})
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(flattened)
    _atomic_write_bytes(csv_path, stream.getvalue().encode("utf-8"), allow_replace=csv_path.exists())


def _prepare_output_directory(output_dir: Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        if not output_dir.is_dir():
            raise TrainingProtocolError("output directory path is not a directory") from None
        if any(output_dir.iterdir()):
            raise TrainingProtocolError("output directory already contains artifacts and is immutable")
    lock_path = output_dir / ".phase1_mirage_trainer.lock"
    try:
        with lock_path.open("x", encoding="utf-8") as handle:
            handle.write("SOURCE_ONLY_TRAINING_IN_PROGRESS\n")
    except FileExistsError as error:
        raise TrainingProtocolError("output directory already has an active or failed trainer lock") from error
    return output_dir, lock_path


def _write_split_receipt(output_dir: Path, config: TrainConfig) -> None:
    _atomic_write_json(
        output_dir / "split_receipt.json",
        {
            "schema": "phase1_mirage_split_receipt_v1",
            "split_sha256": config.split_sha256,
            "loader_roles": sorted(_REQUIRED_LOADER_ROLES),
            "source_only": True,
        },
    )


def _write_resource_receipt(output_dir: Path, *, model: nn.Module, head: nn.Module, device: torch.device) -> None:
    _atomic_write_json(
        output_dir / "resource_receipt.json",
        {
            "schema": "phase1_mirage_resource_receipt_v1",
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "parameter_count": sum(parameter.numel() for parameter in tuple(model.parameters()) + tuple(head.parameters())),
        },
    )


def write_final_checkpoint(
    *,
    output_dir: Path,
    model: MIRAGEEncoder,
    head: MIRAGEOpenHead,
    ema: EMAState,
    swad: SWADAccumulator,
    selected_model_state: Mapping[str, Tensor],
    selected_head_state: Mapping[str, Tensor],
    selection: Mapping[str, object],
    config: TrainConfig,
    epochs_completed: int,
) -> Path:
    """Seal selected deployment weights plus final raw/EMA/SWAD provenance."""

    swad_model_state, swad_head_state, swad_strategy = _swad_state(swad, ema)
    final_ema_model = _clone_state_dict(ema.model)
    final_ema_head = _clone_state_dict(ema.head)
    checkpoint = {
        "schema": "phase1_mirage_trainer_checkpoint_v1",
        "config": asdict(config),
        "config_sha256": _canonical_sha256(asdict(config)),
        "split_sha256": config.split_sha256,
        "git_commit": config.git_commit,
        "epochs_completed": epochs_completed,
        "selection": dict(selection),
        "model_state": dict(selected_model_state),
        "head_state": dict(selected_head_state),
        "last_training_state": {"model": _clone_state_dict(model), "head": _clone_state_dict(head)},
        "ema_state": {"model": final_ema_model, "head": final_ema_head, "updates": ema.updates},
        "swad_state": {
            "model": swad_model_state,
            "head": swad_head_state,
            "count": swad.count,
            "strategy": swad_strategy,
        },
        "rng_state": _capture_rng_state(),
        "state_dict_sha256": {
            "model": _state_dict_sha256(selected_model_state),
            "head": _state_dict_sha256(selected_head_state),
            "ema": _nested_state_sha256(final_ema_model, final_ema_head),
            "swad": _nested_state_sha256(swad_model_state, swad_head_state),
        },
    }
    path = output_dir / "checkpoint.pt"
    _atomic_write_torch(path, checkpoint)
    return path


def write_completion_receipt(
    *,
    output_dir: Path,
    checkpoint_path: Path,
    status: str,
    epochs: int,
    selection: Mapping[str, object],
    swad_strategy: str,
) -> Mapping[str, object]:
    """Validate the sealed checkpoint's bytes and record a terminal receipt."""

    if status != "COMPLETED":
        raise TrainingProtocolError("completion receipt status must be COMPLETED")
    if not checkpoint_path.is_file():
        raise TrainingProtocolError("completion receipt requires an existing checkpoint")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as error:
        raise TrainingProtocolError("completion receipt could not read checkpoint bytes") from error
    if not isinstance(checkpoint, Mapping):
        raise TrainingProtocolError("completion receipt checkpoint payload must be a mapping")
    if int(checkpoint.get("epochs_completed", -1)) != epochs:
        raise TrainingProtocolError("checkpoint epoch count does not match completion receipt")
    checkpoint_selection = checkpoint.get("selection")
    if not isinstance(checkpoint_selection, Mapping):
        raise TrainingProtocolError("checkpoint selection metadata is missing")
    if checkpoint_selection.get("selection_source") != "V_select":
        raise TrainingProtocolError("checkpoint selection source must be V_select")
    if int(checkpoint_selection.get("epoch", -1)) != int(selection["epoch"]):
        raise TrainingProtocolError("checkpoint selected epoch does not match completion receipt")
    if int(selection["epoch"]) < 1 or int(selection["epoch"]) > epochs:
        raise TrainingProtocolError("selected epoch lies outside completed training")
    if selection.get("selection_source") != "V_select":
        raise TrainingProtocolError("checkpoint selection must use V_select")
    receipt = {
        "schema": "phase1_mirage_completion_receipt_v1",
        "status": status,
        "checkpoint_path": checkpoint_path.name,
        "checkpoint_sha256": _file_sha256(checkpoint_path),
        "epochs_completed": epochs,
        "selected_epoch": int(selection["epoch"]),
        "selection_source": "V_select",
        "v_select_known_macro": float(selection["v_select_known_macro"]),
        "v_select_worst_scene": float(selection["v_select_worst_scene"]),
        "swad_strategy": swad_strategy,
    }
    _atomic_write_json(output_dir / "completion_receipt.json", receipt)
    return MappingProxyType(receipt)


def train_fold(config: TrainConfig, loaders: Mapping[str, Iterable[object]], output_dir: Path) -> TrainResult:
    """Train one immutable source fold without target, calibration, or scoring access."""

    if not isinstance(config, TrainConfig):
        raise TypeError("config must be a TrainConfig")
    _validate_loader_schema(loaders)
    _validate_explicit_metadata(config)
    output_dir, lock_path = _prepare_output_directory(Path(output_dir))
    completed = False
    try:
        _write_split_receipt(output_dir, config)
        model, head, optimizer = build_models_and_optimizer(config)
        device = _resolve_device(config)
        _write_resource_receipt(output_dir, model=model, head=head, device=device)
        ema = make_ema_copy(model, head)
        swad = make_swad_accumulator(model, head)
        history: list[dict[str, object]] = []
        selected_model_state: dict[str, Tensor] | None = None
        selected_head_state: dict[str, Tensor] | None = None
        selected_epoch: int | None = None
        proxy_schedule: list[dict[str, int | str]] = []
        for epoch in range(1, config.epochs + 1):
            train_metrics = run_train_epoch(
                model=model,
                head=head,
                ema=ema,
                optimizer=optimizer,
                labeled_loader=loaders["l"],
                unlabeled_loader=loaders["u"],
                epoch=epoch,
                config=config,
            )
            validation = run_source_validation(
                model=model,
                head=head,
                ema=ema,
                v_cal=loaders["v_cal"],
                v_select=loaders["v_select"],
                device=device,
            )
            if config.formal and _stage_for_epoch(epoch, config) == "stabilization":
                update_swad(swad, ema)
            row: dict[str, object] = {
                "epoch": epoch,
                **validation,
                "stage": train_metrics["stage"],
                "optimizer_steps": train_metrics["optimizer_steps"],
                "ema_updates": train_metrics["ema_updates"],
                "ema_updated_after_optimizer_step": train_metrics["ema_updated_after_optimizer_step"],
                "losses": train_metrics["losses"],
                "proxy_schedule": train_metrics["proxy_schedule"],
            }
            history.append(row)
            current_selection = _select_best_epoch(history)
            if selected_epoch != int(current_selection["epoch"]):
                selected_epoch = int(current_selection["epoch"])
                if selected_epoch == epoch:
                    selected_model_state = _clone_state_dict(ema.model)
                    selected_head_state = _clone_state_dict(ema.head)
            proxy_schedule.extend(train_metrics["proxy_schedule"])
            append_epoch_metrics(output_dir / "metrics_epoch.jsonl", history)
        selection = _select_best_epoch(history)
        if selected_model_state is None or selected_head_state is None:
            # The selected epoch must be encountered while its state is live;
            # this branch therefore protects against a future selection rewrite.
            raise TrainingProtocolError("selected V_select epoch state was not captured")
        _atomic_write_json(
            output_dir / "proxy_receipt.json",
            {
                "schema": "phase1_mirage_proxy_receipt_v1",
                "proxy_train_episodes": proxy_schedule,
                "validation_proxy_used": False,
                "selection_source": "V_select",
            },
        )
        checkpoint_path = write_final_checkpoint(
            output_dir=output_dir,
            model=model,
            head=head,
            ema=ema,
            swad=swad,
            selected_model_state=selected_model_state,
            selected_head_state=selected_head_state,
            selection=selection,
            config=config,
            epochs_completed=config.epochs,
        )
        _, _, swad_strategy = _swad_state(swad, ema)
        receipt = write_completion_receipt(
            output_dir=output_dir,
            checkpoint_path=checkpoint_path,
            status="COMPLETED",
            epochs=config.epochs,
            selection=selection,
            swad_strategy=swad_strategy,
        )
        completed = True
        return TrainResult(checkpoint_path=checkpoint_path, completion_receipt=receipt)
    finally:
        if completed and lock_path.exists():
            lock_path.unlink()


__all__ = [
    "EMAState",
    "SWADAccumulator",
    "TrainConfig",
    "TrainResult",
    "TrainingProtocolError",
    "append_epoch_metrics",
    "build_models_and_optimizer",
    "make_ema_copy",
    "make_swad_accumulator",
    "run_source_validation",
    "run_train_epoch",
    "train_fold",
    "update_ema",
    "update_swad",
    "write_completion_receipt",
    "write_final_checkpoint",
]
