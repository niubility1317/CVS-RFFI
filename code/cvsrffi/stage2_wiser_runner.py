"""Support-only WISER-RF training orchestration with no query surface."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import inspect
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from cvsrffi.stage2_wiser_rf import (
    configure_p3_time_first_update,
    configure_progressive_identity_update,
    normalized_l2sp_penalty,
    wiser_dual_supervision_loss,
)
from cvsrffi.stage2_wiser_p3 import (
    cross_fitted_p3_loss,
    identity_fft_diagnostics,
    identity_fft_penalties,
    infer_shared_domain_weights,
    project_auxiliary_gradients,
    shared_domain_manifold_loss,
    stratified_crossfit_indices,
    update_nonnegative_duals,
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


@dataclass(frozen=True)
class WISERP3TrainingConfig:
    """Frozen support-only optimization policy for P3-primary arms N2--N6."""

    fold_count: int = 5
    stage_steps: tuple[int, int, int] = (1500, 2000, 2500)
    diagnostic_interval: int = 100
    risk_rho: float = 2.0
    floor_beta: float = 0.25
    floor_tau: float = 0.1
    learning_rate: float = 1.0e-4
    dual_rate: float = 0.1
    class_risk_epsilon: float = 0.0
    manifold_learning_rate: float = 0.1
    manifold_steps: int = 80
    manifold_l2: float = 0.01
    manifold_weight: float = 0.5
    source_head_weight: float = 1.0
    prototype_weight: float = 0.5
    duplication_weight: float = 0.1
    energy_weight: float = 0.1
    duplication_slack: float = 0.0
    energy_floor: float = 0.0
    interpolation_grid: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25, 0.0)
    seed: int = 713102

    def __post_init__(self) -> None:
        if int(self.fold_count) < 2:
            raise ValueError("P3 fold_count must be at least two")
        if len(self.stage_steps) != 3 or any(int(value) < 0 for value in self.stage_steps):
            raise ValueError("P3 stage_steps must contain three nonnegative values")
        if sum(int(value) for value in self.stage_steps) < 1:
            raise ValueError("P3 training needs at least one optimizer step")
        if int(self.diagnostic_interval) < 1 or int(self.manifold_steps) < 1:
            raise ValueError("P3 diagnostic and manifold steps must be positive")
        finite_nonnegative = (
            self.risk_rho, self.floor_beta, self.dual_rate, self.class_risk_epsilon,
            self.manifold_l2, self.manifold_weight, self.source_head_weight,
            self.prototype_weight, self.duplication_weight, self.energy_weight,
            self.duplication_slack, self.energy_floor,
        )
        if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in finite_nonnegative):
            raise ValueError("P3 weights and constraints must be finite and nonnegative")
        if (
            not math.isfinite(float(self.learning_rate))
            or not math.isfinite(float(self.manifold_learning_rate))
            or float(self.learning_rate) <= 0.0
            or float(self.manifold_learning_rate) <= 0.0
            or not math.isfinite(float(self.floor_tau))
            or float(self.floor_tau) <= 0.0
        ):
            raise ValueError("P3 learning rates and floor_tau must be finite and positive")
        if not self.interpolation_grid or any(
            not math.isfinite(float(alpha)) or not 0.0 <= float(alpha) <= 1.0
            for alpha in self.interpolation_grid
        ):
            raise ValueError("P3 interpolation grid must contain finite [0,1] values")


@dataclass(frozen=True)
class SupportInterpolationResult:
    alpha: float
    state: Mapping[str, torch.Tensor]
    support_metrics: Mapping[str, Any]
    query_rows_used: int = 0


@dataclass(frozen=True)
class WISERP3TrainingAudit:
    arm: str
    optimizer_steps: int
    query_rows_used: int
    stage_audits: tuple[Mapping[str, Any], ...]
    reached_parameter_names: tuple[str, ...]
    final_oof_p3_ba: float
    final_oof_p3_floor: float
    baseline_joint_condition_number: float
    final_joint_condition_number: float
    final_zero_identity_count: int
    final_duals: tuple[float, ...]
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


def _clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _p3_support_inputs(
    model: nn.Module,
    support_iq: torch.Tensor,
    support_labels: torch.Tensor,
    support_tokens: Sequence[str],
    config: WISERP3TrainingConfig,
) -> tuple[torch.Tensor, torch.Tensor, tuple[str, ...], torch.Tensor, tuple[Any, ...]]:
    values = torch.as_tensor(support_iq)
    labels = torch.as_tensor(support_labels, device=values.device, dtype=torch.long).view(-1)
    tokens = tuple(str(token) for token in support_tokens)
    if values.ndim != 3 or tuple(values.shape[1:]) != (2, 256) or values.shape[0] != labels.numel():
        raise ValueError("P3 support IQ must be [rows,2,256] and align with labels")
    if not torch.isfinite(values).all() or len(tokens) != len(labels) or len(set(tokens)) != len(tokens):
        raise ValueError("P3 support IQ and opaque support tokens must be finite and unique")
    expected = torch.arange(6, device=labels.device)
    if not torch.equal(torch.unique(labels, sorted=True), expected):
        raise ValueError("P3 support labels must contain exactly six zero-based classes")
    if any(int((labels == class_id).sum()) < int(config.fold_count) for class_id in range(6)):
        raise ValueError("P3 support requires at least one held-out item per class and fold")
    with torch.inference_mode():
        source_logits, identity = _forward_identity(model, values)
    if source_logits.shape != (len(labels), 6) or identity.shape != (len(labels), 160):
        raise ValueError("P3 requires frozen six-class logits and 160D identity features")
    from cvsrffi.stage2_binova_features import make_fft96

    support_np = np.asarray(values.detach().cpu().tolist(), dtype=np.float32)
    fft = torch.as_tensor(make_fft96(support_np), device=values.device, dtype=torch.float32).detach()
    if fft.shape != (len(labels), 96) or not torch.isfinite(fft).all():
        raise ValueError("P3 fixed support FFT96 geometry drift")
    folds = tuple(
        stratified_crossfit_indices(labels, tokens, fold_count=int(config.fold_count), seed=int(config.seed))
    )
    return values, labels, tokens, fft, folds


def _p3_metrics(
    model: nn.Module,
    values: torch.Tensor,
    labels: torch.Tensor,
    fft: torch.Tensor,
    folds: Sequence[Any],
    baseline_class_risk: torch.Tensor,
    duals: torch.Tensor,
    config: WISERP3TrainingConfig,
) -> dict[str, Any]:
    # The exact D92 fitter performs its own local autograd updates even when
    # these support metrics are detached from model optimization.
    with torch.enable_grad():
        _, identity = _forward_identity(model, values)
        p3 = cross_fitted_p3_loss(
            identity, fft, labels, folds=folds, baseline_class_risk=baseline_class_risk,
            class_duals=duals, epsilon=torch.full_like(duals, float(config.class_risk_epsilon)),
            rho=float(config.risk_rho), beta=float(config.floor_beta), tau=float(config.floor_tau),
        )
        diagnostics = identity_fft_diagnostics(identity, fft, labels)
        accuracies = torch.stack([
            (p3.oof_predictions[labels == class_id] == class_id).float().mean()
            for class_id in range(6)
        ])
    return {
        "oof_p3_ba": float(accuracies.mean()),
        "oof_p3_floor": float(accuracies.min()),
        "oof_mean_risk": float(p3.mean_risk.detach()),
        "oof_class_risk": tuple(float(value.detach()) for value in p3.class_risk),
        "joint_condition_number": diagnostics.joint_condition_number,
        "zero_identity_count": diagnostics.zero_identity_count,
        "cross_covariance_frobenius": diagnostics.cross_covariance_frobenius,
    }


def select_support_safe_interpolation(
    base_state: Mapping[str, torch.Tensor],
    candidate_state: Mapping[str, torch.Tensor],
    *,
    evaluator: Callable[[Mapping[str, torch.Tensor]], Mapping[str, Any] | bool],
    grid: Sequence[float] = (1.0, 0.75, 0.5, 0.25, 0.0),
    trainable_parameter_names: Sequence[str] | None = None,
) -> SupportInterpolationResult:
    """Select the first safe support-only interpolation, otherwise retain base."""

    if set(base_state) != set(candidate_state):
        raise ValueError("interpolation states must have identical members")
    if not grid or any(not math.isfinite(float(alpha)) or not 0.0 <= float(alpha) <= 1.0 for alpha in grid):
        raise ValueError("interpolation grid must contain finite [0,1] values")
    trainable = set(base_state) if trainable_parameter_names is None else set(trainable_parameter_names)
    if not trainable <= set(base_state):
        raise ValueError("interpolation whitelist is absent from state")
    fallback: dict[str, torch.Tensor] = {name: value.detach().clone() for name, value in base_state.items()}
    fallback_metrics: Mapping[str, Any] = {"safe": False}
    for raw_alpha in grid:
        alpha = float(raw_alpha)
        state: dict[str, torch.Tensor] = {}
        for name, base in base_state.items():
            candidate = candidate_state[name]
            if name in trainable and torch.is_floating_point(base):
                state[name] = base.detach() + alpha * (candidate.detach() - base.detach())
            else:
                state[name] = base.detach().clone()
        observed = evaluator(state)
        metrics: Mapping[str, Any] = {"safe": bool(observed)} if isinstance(observed, bool) else observed
        if alpha == 0.0:
            fallback, fallback_metrics = state, metrics
        if bool(metrics.get("safe", False)):
            return SupportInterpolationResult(alpha=alpha, state=state, support_metrics=metrics)
    return SupportInterpolationResult(alpha=0.0, state=fallback, support_metrics=fallback_metrics)


def train_wiser_p3_arm(
    model: nn.Module,
    support_iq: torch.Tensor,
    support_labels: torch.Tensor,
    *,
    support_tokens: Sequence[str],
    source_summary: QuantizedSourceSummary | None,
    arm: str,
    config: WISERP3TrainingConfig,
) -> WISERP3TrainingAudit:
    """Run the P3 arm and refreeze even rejected inputs or failed training."""

    try:
        return _train_wiser_p3_arm_impl(
            model, support_iq, support_labels, support_tokens=support_tokens,
            source_summary=source_summary, arm=arm, config=config,
        )
    finally:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)


def _train_wiser_p3_arm_impl(
    model: nn.Module,
    support_iq: torch.Tensor,
    support_labels: torch.Tensor,
    *,
    support_tokens: Sequence[str],
    source_summary: QuantizedSourceSummary | None,
    arm: str,
    config: WISERP3TrainingConfig,
) -> WISERP3TrainingAudit:
    """Train P3-primary arms from support only, then leave the model refrozen."""

    arm_value = str(arm).upper()
    if arm_value not in {"N2", "N3", "N4", "N5", "N6"}:
        raise ValueError("P3 arm must be N2, N3, N4, N5 or N6")
    values, labels, _, fft, folds = _p3_support_inputs(model, support_iq, support_labels, support_tokens, config)
    needs_summary = arm_value in {"N4", "N5", "N6"}
    if needs_summary:
        if source_summary is None or len(source_summary.class_registry) != 6:
            raise ValueError("N4-N6 require a valid six-class immutable source summary")
        source_points = source_summary.domain_class_points().to(values.device)
        if source_points.ndim != 3 or source_points.shape[1:] != (6, 160):
            raise ValueError("N4-N6 require [domain,6,160] source summary points")
    else:
        source_points = None

    initial_state = _clone_state(model)
    anchors = {name: value.detach().clone() for name, value in model.named_parameters()}
    with torch.enable_grad():
        _, initial_identity = _forward_identity(model, values)
        baseline = cross_fitted_p3_loss(
            initial_identity.detach(), fft, labels, folds=folds,
            baseline_class_risk=torch.zeros(6, device=values.device),
            class_duals=torch.zeros(6, device=values.device), epsilon=torch.zeros(6, device=values.device),
            rho=0.0, beta=0.0, tau=float(config.floor_tau),
        )
        baseline_class_risk = baseline.class_risk.detach()
        baseline_diagnostics = identity_fft_diagnostics(initial_identity, fft, labels)
    shared_weights = None
    if source_points is not None:
        shared_weights = infer_shared_domain_weights(
            initial_identity.detach(), labels, source_points, steps=int(config.manifold_steps),
            learning_rate=float(config.manifold_learning_rate), l2=float(config.manifold_l2),
        ).detach()
    duals = torch.zeros(6, device=values.device)
    reached: set[str] = set()
    rows: list[dict[str, Any]] = []
    optimizer_steps = 0

    def evaluate(state: Mapping[str, torch.Tensor], anchor_metrics: Mapping[str, Any]) -> Mapping[str, Any]:
        model.load_state_dict(state, strict=True)
        model.eval()
        metrics = _p3_metrics(model, values, labels, fft, folds, baseline_class_risk, duals, config)
        metrics["safe"] = bool(
            metrics["oof_p3_ba"] > float(anchor_metrics["oof_p3_ba"])
            and metrics["oof_p3_floor"] >= float(anchor_metrics["oof_p3_floor"])
            and metrics["zero_identity_count"] == 0
            and metrics["joint_condition_number"] <= 2.0 * baseline_diagnostics.joint_condition_number
        )
        return metrics

    def train_branch(
        branch: str, parent_branch: str | None, stage_input: Mapping[str, torch.Tensor], step_count: int,
    ) -> tuple[SupportInterpolationResult, dict[str, Any]]:
        nonlocal optimizer_steps, duals
        model.load_state_dict(stage_input, strict=True)
        update = configure_p3_time_first_update(model, branch=branch, parent_branch=parent_branch)
        trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW([{"params": [parameter], "lr": float(config.learning_rate)} for _, parameter in trainable], weight_decay=0.0)
        anchor_metrics = _p3_metrics(model, values, labels, fft, folds, baseline_class_risk, duals, config)
        projection_audits: list[Mapping[str, float]] = []
        for _ in range(int(step_count)):
            optimizer.zero_grad(set_to_none=True)
            source_logits, identity = _forward_identity(model, values)
            p3 = cross_fitted_p3_loss(
                identity, fft, labels, folds=folds, baseline_class_risk=baseline_class_risk,
                class_duals=(duals if arm_value != "N2" else torch.zeros_like(duals)),
                epsilon=torch.full_like(duals, float(config.class_risk_epsilon)),
                rho=(0.0 if arm_value == "N2" else float(config.risk_rho)),
                beta=(0.0 if arm_value == "N2" else float(config.floor_beta)), tau=float(config.floor_tau),
            )
            manifold = p3.total.new_zeros(())
            if source_points is not None:
                assert shared_weights is not None
                manifold = shared_domain_manifold_loss(identity, labels, source_points, shared_weights)
            if arm_value in {"N5", "N6"}:
                dual = wiser_dual_supervision_loss(source_logits, identity, labels, lambda_proto=1.0)
                auxiliary = float(config.source_head_weight) * dual.source_head + float(config.prototype_weight) * dual.target_proto + float(config.manifold_weight) * manifold
                if arm_value == "N6":
                    duplication, energy = identity_fft_penalties(
                        identity, fft, labels,
                        baseline_cross_covariance_frobenius=baseline_diagnostics.cross_covariance_frobenius,
                        duplication_slack=float(config.duplication_slack), energy_floor=float(config.energy_floor),
                    )
                    auxiliary = auxiliary + float(config.duplication_weight) * duplication + float(config.energy_weight) * energy
                primary_grads = torch.autograd.grad(p3.total, [parameter for _, parameter in trainable], retain_graph=True, allow_unused=True)
                auxiliary_grads = torch.autograd.grad(auxiliary, [parameter for _, parameter in trainable], allow_unused=True)
                projected, projection_audit = project_auxiliary_gradients(primary_grads, auxiliary_grads, reference=[parameter for _, parameter in trainable])
                projection_audits.append(projection_audit)
                for (_, parameter), primary_grad, projected_grad in zip(trainable, primary_grads, projected):
                    parameter.grad = (torch.zeros_like(parameter) if primary_grad is None else primary_grad) + projected_grad
            else:
                total = p3.total + (float(config.manifold_weight) * manifold if arm_value == "N4" else 0.0)
                total.backward()
            for name, parameter in trainable:
                if parameter.grad is not None:
                    if not torch.isfinite(parameter.grad).all():
                        raise RuntimeError(f"P3 gradient became nonfinite: {name}")
                    reached.add(name)
            optimizer.step()
            optimizer_steps += 1
            if arm_value != "N2":
                duals = update_nonnegative_duals(duals, p3.violation.detach(), rate=float(config.dual_rate))
        candidate_state = _clone_state(model)
        selected = select_support_safe_interpolation(
            stage_input, candidate_state, evaluator=lambda state: evaluate(state, anchor_metrics),
            grid=config.interpolation_grid, trainable_parameter_names=update.trainable_parameter_names,
        )
        model.load_state_dict(selected.state, strict=True)
        row = {
            "branch": branch, "parent_branch": parent_branch, "steps": int(step_count),
            "alpha": selected.alpha, "trainable_parameter_names": list(update.trainable_parameter_names),
            "trainable_parameter_count": update.trainable_parameter_count,
            "support_metrics": dict(selected.support_metrics), "query_rows_used": 0,
            "gradient_projection_audits": tuple(projection_audits),
        }
        return selected, row

    try:
        stage1, row = train_branch("stage1_time", None, initial_state, int(config.stage_steps[0]))
        rows.append(row)
        stage2_results: list[tuple[SupportInterpolationResult, dict[str, Any]]] = []
        for branch in ("stage2_time", "stage2_frequency", "stage2_joint"):
            selected, branch_row = train_branch(branch, "stage1_time", stage1.state, int(config.stage_steps[1]))
            rows.append(branch_row)
            stage2_results.append((selected, branch_row))
        chosen_state, chosen_row = min(
            stage2_results,
            key=lambda item: (
                -float(item[0].support_metrics["oof_p3_ba"]),
                -float(item[0].support_metrics["oof_p3_floor"]),
                float(item[0].support_metrics["joint_condition_number"]),
                int(item[1]["trainable_parameter_count"]), item[1]["branch"],
            ),
        )
        stage3, row = train_branch("stage3", str(chosen_row["branch"]), chosen_state.state, int(config.stage_steps[2]))
        rows.append(row)
        final_metrics = dict(stage3.support_metrics)
    finally:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    return WISERP3TrainingAudit(
        arm=arm_value, optimizer_steps=optimizer_steps, query_rows_used=0,
        stage_audits=tuple(rows), reached_parameter_names=tuple(sorted(reached)),
        final_oof_p3_ba=float(final_metrics["oof_p3_ba"]), final_oof_p3_floor=float(final_metrics["oof_p3_floor"]),
        baseline_joint_condition_number=float(baseline_diagnostics.joint_condition_number),
        final_joint_condition_number=float(final_metrics["joint_condition_number"]),
        final_zero_identity_count=int(final_metrics["zero_identity_count"]),
        final_duals=tuple(float(value) for value in duals), config=asdict(config),
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
    "SupportInterpolationResult",
    "WISERP3TrainingAudit",
    "WISERP3TrainingConfig",
    "WISERTrainingAudit",
    "WISERTrainingConfig",
    "predict_wiser_representation_probes",
    "select_support_safe_interpolation",
    "train_wiser_arm",
    "train_wiser_p3_arm",
]
