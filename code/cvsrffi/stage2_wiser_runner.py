"""Support-only WISER-RF training orchestration with no query surface."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import inspect
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from cvsrffi.stage2_wiser_rf import (
    configure_progressive_identity_update,
    normalized_l2sp_penalty,
    wiser_dual_supervision_loss,
)
from cvsrffi.wiser_model_inversion import invert_source_head_iq
from cvsrffi.wiser_source_summary import (
    QuantizedSourceSummary,
    classwise_sliced_wasserstein,
)


@dataclass(frozen=True)
class WISERTrainingConfig:
    stage_steps: tuple[int, int, int] = (1500, 2500, 4000)
    projection_learning_rate: float = 3.0e-4
    late_learning_rate: float = 1.0e-4
    middle_learning_rate: float = 3.0e-5
    early_learning_rate: float = 1.0e-5
    lambda_proto: float = 0.5
    lambda_sp: float = 1.0
    lambda_vsw: float = 0.5
    lambda_inversion: float = 0.25
    prototype_scale: float = 10.0
    num_vsw_projections: int = 32
    inversion_steps: int = 300
    inversion_learning_rate: float = 0.03
    inversion_samples_per_class: int = 2
    seed: int = 0

    def __post_init__(self) -> None:
        if len(self.stage_steps) != 3 or any(int(value) < 0 for value in self.stage_steps):
            raise ValueError("stage_steps must contain three nonnegative values")
        if sum(int(value) for value in self.stage_steps) < 1:
            raise ValueError("WISER training needs at least one optimizer step")
        for value in (
            self.projection_learning_rate,
            self.late_learning_rate,
            self.middle_learning_rate,
            self.early_learning_rate,
        ):
            if float(value) <= 0.0:
                raise ValueError("all WISER learning rates must be positive")


@dataclass(frozen=True)
class WISERTrainingAudit:
    arm: str
    optimizer_steps: int
    query_rows_used: int
    vsw_enabled: bool
    model_inversion_enabled: bool
    stage_audits: tuple[Mapping[str, Any], ...]
    config: Mapping[str, Any]


def _forward_identity(model: nn.Module, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        parameters = inspect.signature(model.forward).parameters
    except (TypeError, ValueError) as exc:
        raise ValueError("cannot inspect WISER model.forward") from exc
    kwargs: dict[str, Any] = {}
    if "return_aux" in parameters:
        kwargs["return_aux"] = True
    for label_name in ("y_tx", "y"):
        if label_name in parameters:
            kwargs[label_name] = None
            break
    outputs = model(values, **kwargs)
    if not isinstance(outputs, Mapping):
        raise ValueError("WISER model must return auxiliary mapping")
    logits = outputs.get("tx_logits", outputs.get("logits"))
    features = outputs.get("z_id")
    if (
        not torch.is_tensor(logits)
        or not torch.is_tensor(features)
        or logits.ndim != 2
        or features.ndim != 2
        or logits.shape[0] != values.shape[0]
        or features.shape[0] != values.shape[0]
    ):
        raise ValueError("WISER model identity output geometry drift")
    return logits, features


def _learning_rate_for(name: str, config: WISERTrainingConfig) -> float:
    if name.startswith(("id_backbone.t1.", "id_backbone.f1.", "id_backbone.time_fuse.", "id_backbone.freq_gate.", "id_backbone.freq_stats_proj.")):
        return float(config.early_learning_rate)
    if name.startswith(("id_backbone.t2.", "id_backbone.f2.")):
        return float(config.middle_learning_rate)
    if name.startswith(("id_backbone.t3.", "id_backbone.f3.")):
        return float(config.late_learning_rate)
    return float(config.projection_learning_rate)


def _optimizer(model: nn.Module, config: WISERTrainingConfig) -> torch.optim.Optimizer:
    groups = [
        {"params": [parameter], "lr": _learning_rate_for(name, config)}
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not groups:
        raise ValueError("WISER stage has no trainable parameters")
    return torch.optim.AdamW(groups, weight_decay=0.0)


def train_wiser_arm(
    model: nn.Module,
    support_iq: torch.Tensor,
    support_labels: torch.Tensor,
    *,
    source_summary: QuantizedSourceSummary | None,
    arm: str,
    config: WISERTrainingConfig,
) -> WISERTrainingAudit:
    """Mutate one fresh checkpoint using legal support only, then refreeze it."""

    arm_value = str(arm).upper()
    if arm_value not in {"A", "B", "C", "ABC"}:
        raise ValueError("WISER arm must be A, B, C or ABC")
    values = torch.as_tensor(support_iq)
    labels = torch.as_tensor(support_labels, dtype=torch.long, device=values.device).view(-1)
    if values.ndim != 3 or values.shape[0] != labels.numel() or not torch.isfinite(values).all():
        raise ValueError("WISER support IQ/labels are invalid")
    classes = torch.unique(labels, sorted=True)
    if not torch.equal(classes, torch.arange(len(classes), device=labels.device)):
        raise ValueError("WISER support labels must be contiguous zero-based")
    if any(int((labels == index).sum()) < 2 for index in range(len(classes))):
        raise ValueError("WISER dual supervision requires K>=2 per class")

    use_vsw = arm_value in {"B", "ABC"}
    use_inversion = arm_value in {"C", "ABC"}
    if use_vsw and source_summary is None:
        raise ValueError("B/ABC requires a quantized source summary")
    if source_summary is not None and len(source_summary.class_registry) != len(classes):
        raise ValueError("source summary/support class registry size drift")

    initial = {name: value.detach().clone() for name, value in model.named_parameters()}
    pseudo_iq = None
    pseudo_labels = None
    if use_inversion:
        inversion = invert_source_head_iq(
            model,
            class_ids=range(len(classes)),
            samples_per_class=int(config.inversion_samples_per_class),
            input_channels=int(values.shape[1]),
            input_length=int(values.shape[2]),
            steps=int(config.inversion_steps),
            learning_rate=float(config.inversion_learning_rate),
            seed=int(config.seed),
        )
        pseudo_iq = inversion.pseudo_iq.to(values.device, values.dtype)
        pseudo_labels = inversion.class_ids.to(values.device)

    source_points = (
        None
        if source_summary is None
        else source_summary.virtual_source_points().to(values.device)
    )
    stage_rows = []
    optimizer_steps = 0
    try:
        for stage, step_count in enumerate(config.stage_steps, start=1):
            if int(step_count) == 0:
                continue
            update = configure_progressive_identity_update(model, stage=stage)
            optimizer = _optimizer(model, config)
            final_metrics: dict[str, float] = {}
            reached_gradients: set[str] = set()
            for step_index in range(int(step_count)):
                optimizer.zero_grad(set_to_none=True)
                source_logits, features = _forward_identity(model, values)
                dual = wiser_dual_supervision_loss(
                    source_logits,
                    features,
                    labels,
                    lambda_proto=float(config.lambda_proto),
                    prototype_scale=float(config.prototype_scale),
                )
                trainable = [
                    (name, parameter)
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad
                ]
                l2sp = normalized_l2sp_penalty(trainable, initial)
                total = dual.total + float(config.lambda_sp) * l2sp
                vsw = total.new_zeros(())
                if use_vsw:
                    assert source_points is not None
                    vsw = classwise_sliced_wasserstein(
                        features,
                        labels,
                        source_points,
                        num_projections=int(config.num_vsw_projections),
                        seed=int(config.seed),
                    )
                    total = total + float(config.lambda_vsw) * vsw
                inversion_ce = total.new_zeros(())
                if use_inversion:
                    assert pseudo_iq is not None and pseudo_labels is not None
                    pseudo_logits, _ = _forward_identity(model, pseudo_iq)
                    inversion_ce = F.cross_entropy(pseudo_logits, pseudo_labels)
                    total = total + float(config.lambda_inversion) * inversion_ce
                if not torch.isfinite(total):
                    raise RuntimeError("WISER support loss became nonfinite")
                total.backward()
                for name, parameter in trainable:
                    if parameter.grad is not None:
                        if not torch.isfinite(parameter.grad).all():
                            raise RuntimeError(f"WISER gradient became nonfinite: {name}")
                        reached_gradients.add(name)
                if not reached_gradients:
                    raise RuntimeError("WISER loss did not reach any trainable parameter")
                optimizer.step()
                optimizer_steps += 1
                final_metrics = {
                    "final_total_loss": float(total.detach().cpu()),
                    "final_source_head_loss": float(dual.source_head.detach().cpu()),
                    "final_target_proto_loss": float(dual.target_proto.detach().cpu()),
                    "final_l2sp": float(l2sp.detach().cpu()),
                    "final_vsw": float(vsw.detach().cpu()),
                    "final_inversion_ce": float(inversion_ce.detach().cpu()),
                }
            stage_rows.append(
                {
                    "stage": stage,
                    "steps": int(step_count),
                    "trainable_parameter_count": update.trainable_parameter_count,
                    "trainable_parameter_names": list(update.trainable_parameter_names),
                    "gradient_reached_parameter_names": sorted(reached_gradients),
                    **final_metrics,
                }
            )
    finally:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    return WISERTrainingAudit(
        arm=arm_value,
        optimizer_steps=optimizer_steps,
        query_rows_used=0,
        vsw_enabled=use_vsw,
        model_inversion_enabled=use_inversion,
        stage_audits=tuple(stage_rows),
        config=asdict(config),
    )


def predict_wiser_representation_probes(
    model: nn.Module,
    support_iq: torch.Tensor,
    support_labels: torch.Tensor,
    query_iq: torch.Tensor,
    *,
    query_tokens: tuple[str, ...],
    source_summary: QuantizedSourceSummary,
    seed: int,
) -> Mapping[str, np.ndarray]:
    """Open unlabeled query only after support adaptation is fully frozen."""

    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("WISER query prediction requires a frozen eval model")
    support = torch.as_tensor(support_iq)
    query = torch.as_tensor(query_iq, device=support.device, dtype=support.dtype)
    labels = torch.as_tensor(support_labels, device=support.device, dtype=torch.long).view(-1)
    if support.ndim != 3 or query.ndim != 3 or support.shape[0] != labels.numel():
        raise ValueError("WISER prediction support/query geometry drift")
    if len(query_tokens) != int(query.shape[0]) or len(set(query_tokens)) != len(query_tokens):
        raise ValueError("WISER query tokens must be unique and align with query rows")
    if len(source_summary.class_registry) != 6:
        raise ValueError("formal WISER P1-P3 probe requires six frozen old classes")

    with torch.inference_mode():
        support_logits, support_features = _forward_identity(model, support)
        query_logits, query_features = _forward_identity(model, query)
    if support_logits.shape[1] != 6 or query_logits.shape[1] != 6:
        raise ValueError("WISER frozen source head class count drift")
    if support_features.shape[1] != 160 or query_features.shape[1] != 160:
        raise ValueError("WISER z_id feature dimension must be 160")
    if source_summary.centers.shape != (6, 160):
        raise ValueError("WISER frozen source prototype geometry drift")

    normalized_query = F.normalize(query_features.float(), dim=1)
    source_centers = F.normalize(
        source_summary.centers.to(query.device, torch.float32), dim=1
    )
    p2_logits = normalized_query @ source_centers.T
    p1_predictions = query_logits.argmax(dim=1)
    p2_predictions = p2_logits.argmax(dim=1)

    support_np = np.asarray(support.detach().cpu().tolist(), dtype=np.float32)
    query_np = np.asarray(query.detach().cpu().tolist(), dtype=np.float32)
    support_identity = np.asarray(
        support_features.detach().cpu().tolist(), dtype=np.float32
    )
    query_identity = np.asarray(query_features.detach().cpu().tolist(), dtype=np.float32)
    labels_np = np.asarray(labels.detach().cpu().tolist(), dtype=np.int64)
    from cvsrffi.stage2_binova_d92 import exact_d92_fit
    from cvsrffi.stage2_binova_features import make_fft96

    support_fft = make_fft96(support_np)
    query_fft = make_fft96(query_np)
    d92 = exact_d92_fit(
        support_identity,
        support_fft,
        labels_np,
        class_ids=range(6),
        old_class_count=6,
        seed=int(seed),
        device=str(query.device),
    )
    p3_logits = np.asarray(d92.score(query_identity, query_fft), dtype=np.float32)
    p3_predictions = np.asarray(d92.predict(query_identity, query_fft), dtype=np.int64)
    return {
        "query_tokens": np.asarray(query_tokens),
        "p1_logits": np.asarray(query_logits.detach().cpu().tolist(), dtype=np.float32),
        "p1_predictions": np.asarray(
            p1_predictions.detach().cpu().tolist(), dtype=np.int64
        ),
        "p2_logits": np.asarray(p2_logits.detach().cpu().tolist(), dtype=np.float32),
        "p2_predictions": np.asarray(
            p2_predictions.detach().cpu().tolist(), dtype=np.int64
        ),
        "p3_logits": p3_logits,
        "p3_predictions": p3_predictions,
        "query_z_id": query_identity,
    }


__all__ = [
    "WISERTrainingAudit",
    "WISERTrainingConfig",
    "predict_wiser_representation_probes",
    "train_wiser_arm",
]
