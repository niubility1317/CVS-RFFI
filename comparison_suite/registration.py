"""Inductive, paired old-domain adaptation and native CIL mechanisms.

Only fitting functions accept support or explicitly authorized source tensors.
Query access is confined to ``score``/``predict`` in evaluation mode. These are
matched-protocol extensions, not claims of reproducing every original dataset:
CSIL preserves expansion/masks/KD/source Fisher, MoPC-HR preserves paper
prototype correction/augmentation and squared hierarchical regularization,
and Orthogonal SEI preserves frozen encoder/new-weight margin calibration.
The source factories must train each method's own base mechanism separately.
"""
from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from paper_reproduction.CSIL.losses import compute_csil_loss
from paper_reproduction.mopc_hr_non_exemplar_cil_sei.algorithm import (
    compute_class_prototypes, correct_old_prototypes, prototype_augmentation,
    mopc_hr_incremental_objective,
)
from paper_reproduction.orthogonal_incremental_sei.losses import incremental_calibration_loss

METHOD_ALIASES = {
    "CSIL": "csil", "MoPC-HR": "mopc_hr", "mopc-hr": "mopc_hr",
    "orthogonal_incremental_sei": "orthogonal", "orthogonal_incremental": "orthogonal",
    "protonet_cda": "protonet", "ProtoNet": "protonet",
}
METHODS = ("csil", "mopc_hr", "orthogonal", "protonet")
Logger = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class RegistrationConfig:
    seed: int = 0
    old_adaptation_epochs: int = 10
    old_adaptation_lr: float = 0.001
    old_adaptation_batch_size: int = 128
    csil_epochs: int = 3
    csil_batch_size: int = 20
    csil_lr: float = 0.01
    csil_momentum: float = 0.9
    csil_l2: float = 0.05
    csil_kd_weight: float = 0.2
    csil_ewc_weight: float = 1.0
    csil_train_fraction: float = 0.6
    csil_old_fingerprint_trainable: bool = True
    mopc_epochs: int = 20
    mopc_batch_size: int = 16
    mopc_lr: float = 0.01
    mopc_momentum: float = 0.9
    mopc_weight_decay: float = 0.0002
    mopc_noise_std: float = 0.05
    mopc_alpha: float = 0.97
    mopc_beta: float = 1.0
    mopc_lambda_max: float = 1.0
    mopc_stage_size: int = 5
    orthogonal_epochs: int = 50
    orthogonal_lr: float = 0.08
    orthogonal_top_k: int = 60
    orthogonal_margin: float = 0.2
    orthogonal_tau_fuse: float = 0.01
    orthogonal_lambda_align: float = 1.6
    extraction_batch_size: int = 128

    def __post_init__(self) -> None:
        for name in ("old_adaptation_epochs", "csil_epochs", "mopc_epochs", "orthogonal_epochs"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in ("old_adaptation_batch_size", "csil_batch_size", "mopc_batch_size",
                     "mopc_stage_size", "extraction_batch_size", "orthogonal_top_k"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 < self.csil_train_fraction <= 1:
            raise ValueError("csil_train_fraction must be in (0,1]")


def _config(value: RegistrationConfig | Mapping[str, Any] | None) -> RegistrationConfig:
    return value if isinstance(value, RegistrationConfig) else RegistrationConfig(**dict(value or {}))


def _method(value: str) -> str:
    value = METHOD_ALIASES.get(value, value.lower())
    if value not in METHODS:
        raise ValueError(f"unknown registration method {value!r}")
    return value


def _embedding(model: nn.Module, method: str, x: torch.Tensor) -> torch.Tensor:
    # Lazy import lets source factories and synthetic tests be loaded independently.
    from comparison_suite.models import embedding
    return embedding(model, method, x)


def _logits(model: nn.Module, method: str, x: torch.Tensor) -> torch.Tensor:
    from comparison_suite.models import classifier_logits
    return classifier_logits(model, method, x)


def _features(model: nn.Module, method: str, x: torch.Tensor, batch_size: int) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return torch.cat([_embedding(model, method, x[start:start + batch_size])
                          for start in range(0, len(x), batch_size)], dim=0)


def _validate_support(x: torch.Tensor, y: torch.Tensor, expected: Sequence[int]) -> None:
    if len(x) == 0 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("nonempty support tensors and one label per sample are required")
    if y.dtype != torch.long:
        raise ValueError("support labels must have dtype torch.long")
    if set(y.detach().cpu().tolist()) != set(expected):
        raise ValueError("support labels must cover exactly the registered compact class IDs")
    if not torch.isfinite(x).all():
        raise ValueError("support IQ contains nonfinite values")


def _classes(values: Sequence[int]) -> tuple[int, ...]:
    result = tuple(int(v) for v in values)
    if len(set(result)) != len(result) or not result or any(v < 0 for v in result):
        raise ValueError("classes must be nonempty, unique, and non-negative")
    return result


def _batches(rows: int, epochs: int, batch_size: int, device: torch.device, seed: int):
    generator = torch.Generator(device=device).manual_seed(int(seed))
    iteration = 0
    for epoch in range(1, epochs + 1):
        order = torch.randperm(rows, generator=generator, device=device)
        for start in range(0, rows, batch_size):
            iteration += 1
            yield epoch, iteration, order[start:start + batch_size]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _bytes(model: nn.Module, extra: Any = None) -> int:
    seen: set[int] = set()
    def count(value: Any) -> int:
        if isinstance(value, torch.Tensor):
            if id(value) in seen:
                return 0
            seen.add(id(value))
            return value.numel() * value.element_size()
        if isinstance(value, Mapping):
            return sum(count(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return sum(count(v) for v in value)
        return 0
    return count(dict(model.state_dict())) + count(extra)


def _record(trace: list[dict[str, Any]], model: nn.Module, terms: Mapping[str, Any],
            callback: Logger | None, start: float, **fields: Any) -> None:
    gradients = [p.grad.detach().norm().square() for p in model.parameters() if p.grad is not None]
    row = {**fields, **{key: float(value.detach()) if isinstance(value, torch.Tensor)
                       else value for key, value in terms.items()},
           "gradient_norm": float(terms["gradient_norm"]) if "gradient_norm" in terms else
                            (float(torch.stack(gradients).sum().sqrt()) if gradients else 0.0),
           "elapsed_seconds": time.perf_counter() - start,
           "source_validation": None, "source_validation_reason": "Phase2 does not select on validation"}
    if not all(torch.isfinite(torch.tensor(v)) for v in row.values() if isinstance(v, float)):
        raise FloatingPointError("nonfinite registration training term")
    trace.append(row)
    if callback:
        callback(dict(row))


@dataclass
class RegistrationState:
    method: str
    model: nn.Module
    classes: tuple[int, ...]
    old_count: int
    prototypes: torch.Tensor | None = None
    base_metadata: dict[str, Any] = field(default_factory=dict)
    loss_trace: list[dict[str, Any]] = field(default_factory=list)
    resource: dict[str, Any] = field(default_factory=dict)
    protocol_metadata: dict[str, Any] = field(default_factory=dict)

    @torch.no_grad()
    def score(self, query_x: torch.Tensor) -> torch.Tensor:
        """Return all registered scores, without query-driven state updates."""
        self.model.eval()
        if self.method == "protonet":
            z = _embedding(self.model, self.method, query_x)
            return -torch.cdist(z, self.prototypes).square()
        if self.method == "orthogonal":
            z = _embedding(self.model, self.method, query_x)
            return F.normalize(z, dim=1) @ F.normalize(self.model.classifier_weight, dim=1).t()
        return _logits(self.model, self.method, query_x)[:, :len(self.classes)]

    @torch.no_grad()
    def predict(self, query_x: torch.Tensor) -> torch.Tensor:
        return self.score(query_x).argmax(1)

    @property
    def metadata(self) -> dict[str, Any]:
        return {**self.protocol_metadata, "resource": dict(self.resource),
                "classes": list(self.classes), "old_count": self.old_count}


def build_registration_base_metadata(
    method: str, base_model: nn.Module, source_x: torch.Tensor, source_y: torch.Tensor,
    *, config: RegistrationConfig | Mapping[str, Any] | None = None,
    source_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute base summaries once, strictly from authorized source training L.

    This function must run before target adaptation. The caller records actual
    source physical IDs/checkpoint provenance; source validation is not fitted.
    """
    method, cfg = _method(method), _config(config)
    model = copy.deepcopy(base_model).to(source_x.device).eval()
    if len(source_x) == 0 or len(source_x) != len(source_y):
        raise ValueError("source training tensors are empty or mismatched")
    details = {"source_role": "L_s", "source_rows": len(source_x),
               "computed_before_target_adaptation": True,
               "source_metadata": dict(source_metadata or {})}
    if method == "csil":
        if not hasattr(model, "fc_bf_fp") or not hasattr(model, "fingerprints"):
            raise ValueError("CSIL requires its native projection and fingerprints")
        for p in model.parameters():
            p.requires_grad_(True)
        # Explicit official-code numerical correction: the author's exp(grad^2)
        # approximation overflows even FP64 for legitimate source-only data.
        # Standard empirical EWC Fisher averages per-example source-label CE
        # squared gradients. This is never advertised as literal author parity.
        model.double()
        parameters = dict(model.named_parameters())
        fisher = {key: torch.zeros_like(value) for key, value in parameters.items()}
        for index in range(len(source_x)):
            model.zero_grad(set_to_none=True)
            loss = F.cross_entropy(_logits(model, "csil", source_x[index:index + 1].double()),
                                   source_y[index:index + 1].long())
            loss.backward()
            for key, value in parameters.items():
                if value.grad is not None:
                    fisher[key].add_(value.grad.detach().square(), alpha=1.0 / len(source_x))
        model.zero_grad(set_to_none=True)
        if not all(torch.isfinite(value).all() for value in fisher.values()):
            raise FloatingPointError("CSIL source empirical Fisher is nonfinite; no fallback permitted")
        details["fisher"] = fisher
        details["fisher_objective"] = "empirical_source_label_CE_grad_squared"
        details["fisher_precision"] = "float64"
        details["fisher_correction_reason"] = "author_exp_squared_gradient_overflow_even_float64"
    elif method == "mopc_hr":
        z = _features(model, method, source_x, cfg.extraction_batch_size)
        prototypes, ids = compute_class_prototypes(z, source_y)
        details.update(old_prototypes=prototypes.detach().clone(), source_class_ids=ids.tolist())
    return details


def _select_old_model(model: nn.Module, method: str, classes: tuple[int, ...]) -> nn.Module:
    indices = torch.tensor(classes, device=next(model.parameters()).device)
    if method == "csil":
        if not hasattr(model, "fc_bf_fp") or not hasattr(model, "fingerprints"):
            raise ValueError("CSIL base must expose fc_bf_fp and fingerprints")
        model.fingerprints = nn.Parameter(model.fingerprints.detach()[indices].clone())
    elif method == "mopc_hr":
        if not isinstance(getattr(model, "fc", None), nn.Linear):
            raise ValueError("MoPC-HR requires its native fc classifier")
        old = model.fc
        head = nn.Linear(old.in_features, len(classes), bias=old.bias is not None).to(old.weight)
        with torch.no_grad():
            head.weight.copy_(old.weight[indices])
            if old.bias is not None:
                head.bias.copy_(old.bias[indices])
        model.fc = head
    elif method == "orthogonal":
        if not hasattr(model, "pseudo_targets") or not hasattr(model, "classifier_weight"):
            raise ValueError("Orthogonal base requires trained pseudo targets and classifier weights")
        weight = model.classifier_weight.detach()[indices].clone()
        if "classifier_weight" in model._buffers:
            del model._buffers["classifier_weight"]
        model.classifier_weight = nn.Parameter(weight)
    return model


def _finish_resource(state: RegistrationState, start: float, device: torch.device,
                     trainable: int, steps_start: int = 0) -> None:
    _sync(device)
    state.resource.update(
        training_seconds=time.perf_counter() - start,
        trainable_parameters=int(trainable),
        total_parameters=sum(p.numel() for p in state.model.parameters()),
        optimizer_steps=len(state.loss_trace) - steps_start,
        resident_state_bytes=_bytes(state.model, [state.prototypes, state.base_metadata]),
        added_transfer_bytes=0, added_transfer_reason="support locally received; no new inter-node transfer",
        peak_cuda_allocated_bytes=(torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None),
        peak_host_memory_bytes=None, peak_host_memory_reason="not instrumented",
        hardware=str(device), inference_seconds=None,
        inference_seconds_reason="measured by common runner on sealed query predictions",
    )


def prepare_registration_state(
    method: str, base_model: nn.Module, old_support_x: torch.Tensor,
    old_support_y: torch.Tensor, old_classes: Sequence[int], *,
    config: RegistrationConfig | Mapping[str, Any] | None = None,
    source_x: torch.Tensor | None = None, source_y: torch.Tensor | None = None,
    metadata: Mapping[str, Any] | None = None,
    base_metadata: Mapping[str, Any] | None = None, logger: Logger | None = None,
) -> RegistrationState:
    """Stage B: only old target support; never loads/fits query.

    CSIL/MoPC old-only SFT is explicitly a matched experiment extension.
    Source summaries/Fisher are fixed before B, not source replay in B.
    """
    method, cfg, classes = _method(method), _config(config), _classes(old_classes)
    _validate_support(old_support_x, old_support_y, range(len(classes)))
    device = old_support_x.device
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    summaries = dict(base_metadata or {})
    if method in ("csil", "mopc_hr") and not summaries:
        if source_x is None or source_y is None:
            raise ValueError(f"{method} requires fixed source base metadata or explicit source L tensors")
        summaries = build_registration_base_metadata(method, base_model, source_x, source_y,
                                                      config=cfg, source_metadata=metadata)
    if method in ("csil", "mopc_hr") and (
            summaries.get("source_role") != "L_s" or
            summaries.get("computed_before_target_adaptation") is not True):
        raise ValueError("registration source summaries must be fixed L_s state before target adaptation")
    if method == "csil":
        if "fisher" not in summaries:
            raise ValueError("CSIL source Fisher is missing; support/identity fallback is forbidden")
        if summaries.get("fisher_objective") != "empirical_source_label_CE_grad_squared":
            raise ValueError("CSIL main corefix requires declared per-example source CE empirical Fisher")
        full_fisher = summaries["fisher"]
        indices = torch.tensor(classes, device=device)
        summaries["fisher"] = {k: (v.to(device)[indices].clone() if k == "fingerprints"
                                  else v.to(device).clone()) for k, v in full_fisher.items()}
    model = _select_old_model(copy.deepcopy(base_model).to(device).eval(), method, classes)
    state = RegistrationState(method, model, classes, len(classes), base_metadata=summaries,
        protocol_metadata={"phase": "B", "query_fitting": False,
                           "old_target_support_only_for_B_training": True,
                           "received_view": "residual_noeq", "config": asdict(cfg),
                           "input_metadata": dict(metadata or {}),
                           "source_base_metadata_fixed_before_B": bool(summaries),
                           "claim_boundary": "native_method_core_matched_inductive_protocol_extension"})
    if method == "protonet":
        for p in model.parameters():
            p.requires_grad_(False)
        z = _features(model, method, old_support_x, cfg.extraction_batch_size)
        state.prototypes, _ = compute_class_prototypes(z, old_support_y)
        state.protocol_metadata["B_method"] = "episodically_trained_ProtoNet_old_support_squared_Euclidean_prototypes"
        _finish_resource(state, start, device, 0)
        return state
    if method == "orthogonal":
        for p in model.parameters():
            p.requires_grad_(False)
        model.classifier_weight.requires_grad_(True)
        z = _features(model, method, old_support_x, cfg.extraction_batch_size)
        proto, _ = compute_class_prototypes(z, old_support_y)
        optimizer = torch.optim.SGD([model.classifier_weight], lr=cfg.orthogonal_lr)
        state.protocol_metadata["B_method"] = "frozen_encoder_old_weight_margin_calibration_protocol_extension"
        for epoch in range(1, cfg.old_adaptation_epochs + 1):
            optimizer.zero_grad(set_to_none=True)
            empty = model.classifier_weight.new_empty((0, model.classifier_weight.shape[1]))
            loss, terms = incremental_calibration_loss(z, old_support_y, empty, model.classifier_weight,
                new_class_ids=torch.arange(len(classes), device=device), prototypes=proto,
                top_k=cfg.orthogonal_top_k, margin=cfg.orthogonal_margin,
                tau_fuse=cfg.orthogonal_tau_fuse, lambda_align=cfg.orthogonal_lambda_align)
            loss.backward()
            _record(state.loss_trace, model, {"loss": loss, **terms}, logger, start,
                    phase="B", method=method, epoch=epoch, learning_rate=cfg.orthogonal_lr)
            optimizer.step()
        trainable = model.classifier_weight.numel()
    else:
        for p in model.parameters():
            p.requires_grad_(True)
        # Keep BN statistics fixed; K=1 support is valid and query batch independent.
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.old_adaptation_lr)
        state.protocol_metadata["B_method"] = "old_target_support_only_SFT_protocol_extension"
        before_z = _features(model, method, old_support_x, cfg.extraction_batch_size)
        for epoch, iteration, indices in _batches(len(old_support_x), cfg.old_adaptation_epochs,
                cfg.old_adaptation_batch_size, device, cfg.seed):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(_logits(model, method, old_support_x[indices]), old_support_y[indices])
            loss.backward()
            _record(state.loss_trace, model, {"loss": loss, "cross_entropy": loss}, logger, start,
                    phase="B", method=method, epoch=epoch, iteration=iteration,
                    learning_rate=cfg.old_adaptation_lr)
            optimizer.step()
        if method == "mopc_hr":
            source_ids = summaries.get("source_class_ids")
            if source_ids is None or "old_prototypes" not in summaries:
                raise ValueError("MoPC-HR source class prototypes are missing")
            row_ids = torch.tensor([source_ids.index(v) for v in classes], device=device)
            old_prototypes = summaries["old_prototypes"].to(device)[row_ids].clone()
            before_proto, _ = compute_class_prototypes(before_z, old_support_y)
            after_proto, _ = compute_class_prototypes(
                _features(model, method, old_support_x, cfg.extraction_batch_size), old_support_y)
            state.prototypes = correct_old_prototypes(old_prototypes, before_proto, after_proto,
                                                       alpha=cfg.mopc_alpha)
            state.protocol_metadata["B_prototype_correction"] = "paper_cosine_old_support_feature_drift_extension"
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.eval()
    _finish_resource(state, start, device, trainable)
    return state


def _csil_register(state: RegistrationState, x: torch.Tensor, y: torch.Tensor,
                   cfg: RegistrationConfig, logger: Logger | None, start: float) -> None:
    model = state.model
    teacher = copy.deepcopy(model).eval()
    old_count = state.old_count
    previous = {k: p.detach().clone() for k, p in teacher.named_parameters()}
    fisher = state.base_metadata["fisher"]
    if set(previous) != set(fisher):
        raise ValueError("CSIL source Fisher parameter names do not match native model")
    for k, p in previous.items():
        if fisher[k].shape != p.shape or not torch.isfinite(fisher[k]).all():
            raise ValueError(f"CSIL Fisher mismatch/nonfinite {k}")
    old_projection = model.fc_bf_fp
    old_dim = old_projection.out_features
    added_dim = len(state.classes) - old_count
    expanded = nn.Linear(old_projection.in_features, old_dim + added_dim,
                         bias=old_projection.bias is not None).to(old_projection.weight)
    generator = torch.Generator(device=x.device).manual_seed(cfg.seed + 71)
    with torch.no_grad():
        nn.init.kaiming_uniform_(expanded.weight, a=5 ** 0.5, generator=generator)
        expanded.weight[:old_dim].copy_(old_projection.weight)
        if expanded.bias is not None:
            expanded.bias[:old_dim].copy_(old_projection.bias)
            expanded.bias[old_dim:].zero_()
    model.fc_bf_fp = expanded
    fingerprint = model.fingerprints.new_zeros((len(state.classes), old_dim + added_dim))
    fingerprint[:old_count, :old_dim] = model.fingerprints.detach()
    with torch.no_grad():
        # CSIL embedding() includes fc_bf_fp; new projection consumes raw encoder.
        new_feature = expanded(model.encoder(x))[:, old_dim:]
        means, _ = compute_class_prototypes(new_feature, y)
        fingerprint[old_count:, old_dim:] = F.normalize(means, dim=1)
    model.fingerprints = nn.Parameter(fingerprint)
    for p in model.parameters():
        p.requires_grad_(True)
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    # Paper masks prohibit modifying the inherited encoder and old projection.
    masks = {k: torch.zeros_like(p) for k, p in model.named_parameters()}
    masks["fc_bf_fp.weight"][old_dim:] = 1
    if model.fc_bf_fp.bias is not None:
        masks["fc_bf_fp.bias"][old_dim:] = 1
    masks["fingerprints"][old_count:, old_dim:] = 1
    if cfg.csil_old_fingerprint_trainable:
        masks["fingerprints"][:old_count, :old_dim] = 1
    gen = torch.Generator(device=x.device).manual_seed(cfg.seed + 17)
    selected = []
    for label in torch.unique(y):
        rows = (y == label).nonzero().flatten()
        order = rows[torch.randperm(len(rows), generator=gen, device=x.device)]
        selected.append(order[:max(1, int(cfg.csil_train_fraction * len(rows)))])
    train_indices = torch.cat(selected)
    velocity = {k: torch.zeros_like(p) for k, p in model.named_parameters()}
    for epoch, iteration, rows in _batches(len(train_indices), cfg.csil_epochs, cfg.csil_batch_size,
                                          x.device, cfg.seed + 83):
        indices = train_indices[rows]
        model.zero_grad(set_to_none=True)
        logits = _logits(model, "csil", x[indices])
        with torch.no_grad():
            teacher_logits = _logits(teacher, "csil", x[indices])
        terms = compute_csil_loss(logits=logits, labels=y[indices],
            current_old_response=logits[:, :old_count], previous_old_response=teacher_logits,
            params=dict(model.named_parameters()), previous_params=previous, fisher=fisher,
            kd_weight=cfg.csil_kd_weight, ewc_weight=cfg.csil_ewc_weight)
        terms.total.backward()
        learning_rate = cfg.csil_lr / (1 + 0.01 * iteration)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if parameter.grad is None:
                    continue
                update = (parameter.grad + 2 * cfg.csil_l2 * parameter) * masks[name]
                parameter.grad.mul_(masks[name])
                velocity[name].mul_(cfg.csil_momentum).add_(update)
                parameter.add_(velocity[name], alpha=-learning_rate)
        _record(state.loss_trace, model,
                {"loss": terms.total, "cross_entropy": terms.cross_entropy,
                 "knowledge_distillation": terms.knowledge_distillation, "ewc": terms.ewc},
                logger, start, phase="C", method="csil", epoch=epoch, iteration=iteration,
                learning_rate=learning_rate, kd_weight=cfg.csil_kd_weight, ewc_weight=cfg.csil_ewc_weight)
    state.resource.update(trainable_mask_parameters=sum(int(m.sum()) for m in masks.values()),
                          autograd_parameter_elements=sum(p.numel() for p in model.parameters() if p.requires_grad),
                          trainable_parameter_definition="elements permitted by optimizer masks; autograd surface separately reported",
                          new_support_rows_train=len(train_indices))
    state.protocol_metadata.update(C_method="CSIL_separated_channels_masked_SGDM_KD_EWC",
        csil_version="official_repo_corefix_empirical_Fisher_matched_IQ_extension", source_fisher_preserved=True,
        fisher_objective=state.base_metadata.get("fisher_objective"),
        fisher_correction="standard_empirical_EWC_source_per_example_CE_squared_gradient",
        official_fingerprint_mask_corefix=cfg.csil_old_fingerprint_trainable,
        small_K_adapter="stratified_60_percent_min_one_retained_tail",
        old_fingerprint_trainable=cfg.csil_old_fingerprint_trainable)


def _mopc_register(state: RegistrationState, x: torch.Tensor, y: torch.Tensor,
                   cfg: RegistrationConfig, logger: Logger | None, start: float) -> None:
    model = state.model
    old = model.fc
    expanded = nn.Linear(old.in_features, len(state.classes), bias=old.bias is not None).to(old.weight)
    generator = torch.Generator(device=x.device).manual_seed(cfg.seed + 313)
    with torch.no_grad():
        nn.init.kaiming_uniform_(expanded.weight, a=5 ** 0.5, generator=generator)
        if expanded.bias is not None:
            bound = old.in_features ** -0.5
            expanded.bias.uniform_(-bound, bound, generator=generator)
        expanded.weight[:state.old_count].copy_(old.weight)
        if old.bias is not None:
            expanded.bias[:state.old_count].copy_(old.bias)
    model.fc = expanded
    for p in model.parameters():
        p.requires_grad_(True)
    historical = state.prototypes.detach().clone()
    gen = torch.Generator(device=x.device).manual_seed(cfg.seed + 991)
    stages = []
    for stage_index, begin in enumerate(range(state.old_count, len(state.classes), cfg.mopc_stage_size), 1):
        end = min(begin + cfg.mopc_stage_size, len(state.classes))
        stage_rows = ((y >= begin) & (y < end)).nonzero().flatten()
        reference = copy.deepcopy(model).eval()
        previous = {k: p.detach().clone() for k, p in model.named_parameters()}
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.mopc_lr,
                                    momentum=cfg.mopc_momentum, weight_decay=cfg.mopc_weight_decay)
        for epoch, iteration, rows in _batches(len(stage_rows), cfg.mopc_epochs, cfg.mopc_batch_size,
                                              x.device, cfg.seed + 100003 * stage_index):
            indices = stage_rows[rows]
            optimizer.zero_grad(set_to_none=True)
            logits = _logits(model, "mopc_hr", x[indices])[:, :end]
            augmented, augmented_y = prototype_augmentation(historical,
                torch.arange(begin, device=x.device), num_samples=cfg.mopc_batch_size,
                noise_std=cfg.mopc_noise_std, generator=gen)
            proto_logits = model.fc(augmented)[:, :end]
            terms = mopc_hr_incremental_objective(logits, y[indices], proto_logits, augmented_y,
                dict(model.named_parameters()), previous, beta=cfg.mopc_beta, lambda_max=cfg.mopc_lambda_max)
            terms.total.backward()
            _record(state.loss_trace, model,
                {"loss": terms.total, "cross_entropy": terms.cross_entropy,
                 "prototype_augmentation": terms.prototype_augmentation,
                 "hierarchical_regularization": terms.hierarchical_regularization},
                logger, start, phase="C", method="mopc_hr", stage=stage_index, epoch=epoch,
                iteration=iteration, learning_rate=cfg.mopc_lr, beta=cfg.mopc_beta,
                lambda_max=cfg.mopc_lambda_max, registered_count=end)
            optimizer.step()
        previous_z = _features(reference, "mopc_hr", x[stage_rows], cfg.extraction_batch_size)
        current_z = _features(model, "mopc_hr", x[stage_rows], cfg.extraction_batch_size)
        previous_new, _ = compute_class_prototypes(previous_z, y[stage_rows])
        current_new, _ = compute_class_prototypes(current_z, y[stage_rows])
        historical = torch.cat([correct_old_prototypes(historical, previous_new, current_new,
                                   alpha=cfg.mopc_alpha, similarity_mode="paper_cosine"), current_new])
        stages.append({"new_count": end - begin, "registered_count": end,
                       "prototype_correction_applied": True})
    state.prototypes = historical
    state.protocol_metadata.update(C_method="MoPC_HR_paper_CE_protoaugmentation_squared_layerHR",
        mopc_version="paper_equations_7_to_22", incremental_stages=stages,
        class_order=list(state.classes[state.old_count:]), class_order_frozen_before_query=True,
        prototype_correction_consumed_by_later_stages=len(stages) > 1,
        final_decision="trained_classifier_all_registered_logits",
        claim_note="stage_size_5_ordered_arrival_matched_extension; final_prototype_correction_not_classifier_replacement")


def _orthogonal_register(state: RegistrationState, x: torch.Tensor, y: torch.Tensor,
                         cfg: RegistrationConfig, logger: Logger | None, start: float) -> None:
    model = state.model
    for p in model.parameters():
        p.requires_grad_(False)
    features = _features(model, "orthogonal", x, cfg.extraction_batch_size)
    prototypes, class_ids = compute_class_prototypes(features, y)
    old_weights = model.classifier_weight.detach().clone()
    new_weights = nn.Parameter(F.normalize(prototypes, dim=1))
    optimizer = torch.optim.SGD([new_weights], lr=cfg.orthogonal_lr)
    for epoch in range(1, cfg.orthogonal_epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, terms = incremental_calibration_loss(features, y, old_weights, new_weights,
            new_class_ids=class_ids, prototypes=prototypes, top_k=cfg.orthogonal_top_k,
            margin=cfg.orthogonal_margin, tau_fuse=cfg.orthogonal_tau_fuse,
            lambda_align=cfg.orthogonal_lambda_align)
        loss.backward()
        grad = float(new_weights.grad.norm())
        optimizer.step()
        _record(state.loss_trace, model, {"loss": loss, **terms, "gradient_norm": grad}, logger, start,
                phase="C", method="orthogonal", epoch=epoch, learning_rate=cfg.orthogonal_lr,
                lambda_align=cfg.orthogonal_lambda_align)
        state.loss_trace[-1]["gradient_norm"] = grad
    model.classifier_weight = nn.Parameter(torch.cat([old_weights, new_weights.detach()]), requires_grad=False)
    state.resource["new_weight_trainable_parameters"] = new_weights.numel()
    state.protocol_metadata.update(C_method="OSC_FSCIL_frozen_encoder_new_weight_margin_prototype_alignment",
        source_pseudo_targets_preserved=True, inherited_old_weights_unchanged=True,
        final_decision="cosine_all_registered_weights")


def register_new_classes(
    old_state: RegistrationState, new_support_x: torch.Tensor, new_support_y: torch.Tensor,
    new_classes: Sequence[int], *, config: RegistrationConfig | Mapping[str, Any] | None = None,
    logger: Logger | None = None,
) -> RegistrationState:
    """Stage C inherits the complete B state; caller-owned B remains unchanged.

    New support labels are compact IDs starting at B registered class count.
    No source reread, query fit, query class quota or batch reranking occurs.
    """
    cfg = _config(config or old_state.protocol_metadata.get("config"))
    if not new_classes:
        return copy.deepcopy(old_state)
    classes = _classes(new_classes)
    if set(classes) & set(old_state.classes):
        raise ValueError("new and existing registered classes must be disjoint")
    begin = len(old_state.classes)
    _validate_support(new_support_x, new_support_y, range(begin, begin + len(classes)))
    state = copy.deepcopy(old_state)
    state.old_count = begin
    state.classes += classes
    state.protocol_metadata.update(phase="C", C_inherits_B_state=True,
                                   independent_refit=False, new_support_rows=len(new_support_x),
                                   config=asdict(cfg))
    device = new_support_x.device
    state.model.to(device).eval()
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    steps_start = len(state.loss_trace)
    if state.method == "protonet":
        new_z = _features(state.model, state.method, new_support_x, cfg.extraction_batch_size)
        prototypes, _ = compute_class_prototypes(new_z, new_support_y)
        state.prototypes = torch.cat([state.prototypes, prototypes])
        state.protocol_metadata.update(C_method="episodically_trained_ProtoNet_new_support_squared_Euclidean_prototypes",
                                       inherited_old_prototypes_unchanged=True)
        trainable = 0
    elif state.method == "csil":
        _csil_register(state, new_support_x, new_support_y, cfg, logger, start)
        trainable = state.resource["trainable_mask_parameters"]
    elif state.method == "mopc_hr":
        _mopc_register(state, new_support_x, new_support_y, cfg, logger, start)
        trainable = sum(p.numel() for p in state.model.parameters() if p.requires_grad)
    else:
        _orthogonal_register(state, new_support_x, new_support_y, cfg, logger, start)
        trainable = state.resource["new_weight_trainable_parameters"]
    state.model.eval()
    _finish_resource(state, start, device, trainable, steps_start)
    return state
