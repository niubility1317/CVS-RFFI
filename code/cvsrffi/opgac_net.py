"""OPGAC-Net utilities for frozen-backbone spaceborne CVS-RFFI Stage2.

The module implements a support-only prototype-Gaussian calibration layer. It
never consumes target query samples for registration, threshold calibration, or
online adaptation; prediction is a stateless forward pass over a fixed memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .spaceborne_fewshot import PredictionResult, PrototypeSet, UNKNOWN_LABEL, normalize_rows, validate_stage2_protocol


DECISION_OLD = "old_class"
DECISION_NEW = "new_class"
DECISION_UNKNOWN = "unknown"
DECISION_AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class OPGACConfig:
    """Configuration for the first deployable OPGAC-Net implementation."""

    feature_dim: int
    context_dim: int = 128
    quality_dim: int = 0
    stats_dim: int = 6
    hidden_dim: int = 128
    low_rank: int = 4
    eps: float = 1e-5
    normalize_features: bool = True
    old_shrinkage_kappa: float = 3.0
    new_cov_prior_k: int = 3
    new_mean_prior_k: int = 3
    new_shrinkage_nu: float = 10.0
    cov_shrinkage_nu: float = 20.0
    threshold_shrinkage_nu: float = 10.0
    min_variance: float = 1e-4
    max_variance_scale: float = 4.0
    min_variance_scale: float = 0.25
    min_threshold: float = 1e-3
    max_threshold: float = 1.0e6
    old_new_margin: float = 0.10
    top2_margin: float = 0.02
    overlap_margin: float = 0.20
    ambiguous_on_overlap: bool = True
    default_component_threshold_quantile: float = 0.95
    default_class_threshold: float = 3.0
    max_alpha: float = 0.35
    drift_alarm_shift: float = 0.75

    def __post_init__(self) -> None:
        if self.feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if self.low_rank <= 0:
            raise ValueError("low_rank must be positive")


@dataclass
class GaussianClassState:
    """Multi-component Gaussian state for one old or seen-new transmitter."""

    class_id: int
    group: str
    means: torch.Tensor
    diag_vars: torch.Tensor
    weights: torch.Tensor
    component_thresholds: torch.Tensor
    class_threshold: float
    energy_median: float = 0.0
    energy_iqr: float = 1.0
    support_count: int = 0
    version: int = 1
    lifecycle: str = "confirmed"
    metadata: dict = field(default_factory=dict)

    def clone(self) -> "GaussianClassState":
        return GaussianClassState(
            class_id=int(self.class_id),
            group=str(self.group),
            means=self.means.detach().clone(),
            diag_vars=self.diag_vars.detach().clone(),
            weights=self.weights.detach().clone(),
            component_thresholds=self.component_thresholds.detach().clone(),
            class_threshold=float(self.class_threshold),
            energy_median=float(self.energy_median),
            energy_iqr=float(self.energy_iqr),
            support_count=int(self.support_count),
            version=int(self.version),
            lifecycle=str(self.lifecycle),
            metadata=dict(self.metadata),
        )

    @property
    def device(self) -> torch.device:
        return self.means.device

    @property
    def dim(self) -> int:
        return int(self.means.shape[-1])

    def validate(self, feature_dim: int) -> None:
        if self.means.ndim != 2 or self.means.size(1) != int(feature_dim):
            raise ValueError(f"class {self.class_id} means must be [K,{feature_dim}]")
        if self.diag_vars.shape != self.means.shape:
            raise ValueError(f"class {self.class_id} diag_vars must match means")
        if self.weights.ndim != 1 or self.weights.numel() != self.means.size(0):
            raise ValueError(f"class {self.class_id} weights must be [K]")
        if self.component_thresholds.ndim != 1 or self.component_thresholds.numel() != self.means.size(0):
            raise ValueError(f"class {self.class_id} thresholds must be [K]")
        if self.group not in {"old", "seen_new"}:
            raise ValueError(f"class {self.class_id} has unsupported group={self.group!r}")


@dataclass
class OPGACMemory:
    """Read-mostly orbit memory with immutable ground old states and mutable target states."""

    old_states: dict[int, GaussianClassState]
    new_states: dict[int, GaussianClassState] = field(default_factory=dict)
    ground_old_states: dict[int, GaussianClassState] = field(default_factory=dict)
    domain_context: torch.Tensor | None = None
    uncertainty: float = 1.0
    update_log: list[dict] = field(default_factory=list)
    version: int = 1

    def __post_init__(self) -> None:
        if not self.ground_old_states:
            self.ground_old_states = {int(k): v.clone() for k, v in self.old_states.items()}

    def clone(self) -> "OPGACMemory":
        return OPGACMemory(
            old_states={int(k): v.clone() for k, v in self.old_states.items()},
            new_states={int(k): v.clone() for k, v in self.new_states.items()},
            ground_old_states={int(k): v.clone() for k, v in self.ground_old_states.items()},
            domain_context=None if self.domain_context is None else self.domain_context.detach().clone(),
            uncertainty=float(self.uncertainty),
            update_log=[dict(row) for row in self.update_log],
            version=int(self.version),
        )

    def all_states(self) -> dict[int, GaussianClassState]:
        out = {int(k): v for k, v in self.old_states.items()}
        out.update({int(k): v for k, v in self.new_states.items()})
        return out


@dataclass
class OPGACPrediction:
    decisions: list[str]
    predicted_labels: torch.Tensor
    accepted: torch.Tensor
    best_old_labels: torch.Tensor
    best_new_labels: torch.Tensor
    old_scores: torch.Tensor
    new_scores: torch.Tensor
    best_component_d2: torch.Tensor
    margin_old_new: torch.Tensor
    margin_top2: torch.Tensor
    reject_reasons: list[list[str]]
    diagnostics: dict[str, torch.Tensor] = field(default_factory=dict)


class FixedFeatureTransform(nn.Module):
    """Frozen feature transform `T` for centering, projection and whitening."""

    def __init__(
        self,
        *,
        mean: torch.Tensor,
        projection: torch.Tensor,
        scale: torch.Tensor | None = None,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        mean_t = torch.as_tensor(mean, dtype=torch.float32).view(1, -1)
        proj_t = torch.as_tensor(projection, dtype=torch.float32)
        if proj_t.ndim != 2 or proj_t.size(0) != mean_t.size(1):
            raise ValueError("projection must be [input_dim, output_dim] and match mean")
        if scale is None:
            scale_t = torch.ones(proj_t.size(1), dtype=torch.float32)
        else:
            scale_t = torch.as_tensor(scale, dtype=torch.float32).view(-1)
            if scale_t.numel() != proj_t.size(1):
                raise ValueError("scale must match projection output dim")
        self.register_buffer("mean", mean_t)
        self.register_buffer("projection", proj_t)
        self.register_buffer("scale", scale_t.clamp_min(1e-12))
        self.normalize = bool(normalize)

    @property
    def input_dim(self) -> int:
        return int(self.projection.size(0))

    @property
    def output_dim(self) -> int:
        return int(self.projection.size(1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(features, dtype=torch.float32, device=self.mean.device)
        if x.ndim == 1:
            x = x.view(1, -1)
        if x.ndim != 2 or x.size(1) != self.input_dim:
            raise ValueError(f"expected [N,{self.input_dim}] features, got shape={tuple(x.shape)}")
        out = (x - self.mean) @ self.projection
        out = out / self.scale.view(1, -1)
        return normalize_rows(out) if self.normalize else out


def fit_fixed_feature_transform(
    features: torch.Tensor,
    *,
    output_dim: int,
    whitening: bool = True,
    normalize: bool = True,
    eps: float = 1e-6,
) -> FixedFeatureTransform:
    """Fit a deterministic ground-side PCA/whitening transform."""

    x = torch.as_tensor(features, dtype=torch.float32)
    if x.ndim != 2:
        raise ValueError(f"expected [N,D] features, got shape={tuple(x.shape)}")
    rank = max(1, min(int(output_dim), int(x.size(0)), int(x.size(1))))
    mean = x.mean(dim=0, keepdim=True)
    centered = x - mean
    try:
        _, s, vh = torch.linalg.svd(centered, full_matrices=False)
        projection = vh[:rank].T.contiguous()
        denom = (s[:rank] / max(1.0, float(x.size(0) - 1)) ** 0.5).clamp_min(eps)
    except RuntimeError:
        q, _ = torch.linalg.qr(centered.T, mode="reduced")
        projection = q[:, :rank].contiguous()
        denom = torch.ones(rank, dtype=x.dtype)
    scale = denom if whitening else torch.ones_like(denom)
    return FixedFeatureTransform(mean=mean.squeeze(0), projection=projection, scale=scale, normalize=normalize)


def _as_feature_matrix(features: torch.Tensor, feature_dim: int, *, normalize: bool = True) -> torch.Tensor:
    out = torch.as_tensor(features, dtype=torch.float32)
    if out.ndim == 1:
        out = out.view(1, -1)
    if out.ndim != 2 or out.size(1) != int(feature_dim):
        raise ValueError(f"expected feature matrix [N,{feature_dim}], got shape={tuple(out.shape)}")
    return normalize_rows(out) if normalize else out


def _as_labels(labels: torch.Tensor | Iterable[int], device: torch.device | None = None) -> torch.Tensor:
    out = torch.as_tensor(labels, dtype=torch.long, device=device)
    if out.ndim != 1:
        raise ValueError(f"expected label vector [N], got shape={tuple(out.shape)}")
    return out


def _safe_iqr(values: torch.Tensor, eps: float) -> torch.Tensor:
    if values.numel() < 4:
        return values.std(unbiased=False).clamp_min(eps)
    q75 = torch.quantile(values.float(), 0.75)
    q25 = torch.quantile(values.float(), 0.25)
    return (q75 - q25).clamp_min(eps)


def _component_nll(features: torch.Tensor, state: GaussianClassState, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.as_tensor(features, dtype=torch.float32, device=state.device)
    means = state.means.to(x.device, x.dtype)
    diag = state.diag_vars.to(x.device, x.dtype).clamp_min(eps)
    weights = state.weights.to(x.device, x.dtype).clamp_min(eps)
    weights = weights / weights.sum().clamp_min(eps)
    diff = x[:, None, :] - means[None, :, :]
    d2 = (diff.pow(2) / diag[None, :, :]).sum(dim=-1)
    logdet = torch.log(diag).sum(dim=-1)
    log_scores = torch.log(weights).view(1, -1) - 0.5 * (d2 + logdet.view(1, -1))
    nll = -torch.logsumexp(log_scores, dim=1)
    return nll, d2


def _state_from_single_component(
    *,
    class_id: int,
    group: str,
    mean: torch.Tensor,
    diag_var: torch.Tensor,
    threshold: float,
    class_threshold: float,
    count: int,
    metadata: Mapping | None = None,
) -> GaussianClassState:
    mean = torch.as_tensor(mean, dtype=torch.float32).view(1, -1)
    diag = torch.as_tensor(diag_var, dtype=torch.float32, device=mean.device).view(1, -1).clamp_min(1e-8)
    return GaussianClassState(
        class_id=int(class_id),
        group=str(group),
        means=mean,
        diag_vars=diag,
        weights=torch.ones(1, dtype=torch.float32, device=mean.device),
        component_thresholds=torch.tensor([float(threshold)], dtype=torch.float32, device=mean.device),
        class_threshold=float(class_threshold),
        support_count=int(count),
        metadata=dict(metadata or {}),
    )


def build_old_memory_from_prototypes(
    prototypes: PrototypeSet,
    *,
    config: OPGACConfig,
) -> OPGACMemory:
    """Convert a ground PrototypeSet into an old-class Gaussian memory.

    The current PrototypeSet stores one vector per class; multi-component ground
    banks can be supplied directly with `GaussianClassState` objects.
    """

    vectors = _as_feature_matrix(prototypes.vectors, config.feature_dim, normalize=config.normalize_features)
    labels = _as_labels(prototypes.labels, device=vectors.device)
    counts = torch.as_tensor(prototypes.counts, dtype=torch.long, device=vectors.device)
    diag_meta = prototypes.metadata.get("diag_var")
    if torch.is_tensor(diag_meta):
        diag_meta = torch.as_tensor(diag_meta, dtype=torch.float32, device=vectors.device)
    else:
        diag_meta = torch.full_like(vectors, float(config.min_variance))
    maha_meta = prototypes.metadata.get("mahalanobis_thresholds")
    if torch.is_tensor(maha_meta):
        maha_meta = torch.as_tensor(maha_meta, dtype=torch.float32, device=vectors.device)
    else:
        maha_meta = torch.full((vectors.size(0),), float(config.default_class_threshold), dtype=torch.float32, device=vectors.device)
    states: dict[int, GaussianClassState] = {}
    for idx, label in enumerate(labels.detach().cpu().tolist()):
        state = _state_from_single_component(
            class_id=int(label),
            group="old",
            mean=vectors[idx],
            diag_var=diag_meta[idx].clamp_min(config.min_variance),
            threshold=float(maha_meta[idx].item()),
            class_threshold=float(config.default_class_threshold),
            count=int(counts[idx].item()) if idx < counts.numel() else 0,
            metadata={"source": "ground_prototype_set", "component_count": 1},
        )
        state.validate(config.feature_dim)
        states[int(label)] = state
    return OPGACMemory(old_states=states)


class DeepSetContextEncoder(nn.Module):
    """Permutation-invariant support-only target context encoder."""

    def __init__(self, config: OPGACConfig) -> None:
        super().__init__()
        self.config = config
        input_dim = int(config.feature_dim + config.quality_dim + config.stats_dim)
        hidden = int(config.hidden_dim)
        self.psi = nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.rho = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, int(config.context_dim)))
        self.uncertainty = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(
        self,
        support_features: torch.Tensor,
        *,
        quality_features: torch.Tensor | None = None,
        support_stats: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = _as_feature_matrix(support_features, self.config.feature_dim, normalize=self.config.normalize_features)
        if quality_features is None:
            q = torch.zeros(x.size(0), self.config.quality_dim, dtype=x.dtype, device=x.device)
        else:
            q = torch.as_tensor(quality_features, dtype=x.dtype, device=x.device)
        if support_stats is None:
            stats = torch.zeros(x.size(0), self.config.stats_dim, dtype=x.dtype, device=x.device)
        else:
            stats = torch.as_tensor(support_stats, dtype=x.dtype, device=x.device)
        if q.ndim != 2 or q.size(0) != x.size(0) or q.size(1) != self.config.quality_dim:
            raise ValueError("quality_features must be [N, quality_dim]")
        if stats.ndim != 2 or stats.size(0) != x.size(0) or stats.size(1) != self.config.stats_dim:
            raise ValueError("support_stats must be [N, stats_dim]")
        pooled = self.psi(torch.cat([x, q, stats], dim=1)).mean(dim=0, keepdim=True)
        context = self.rho(pooled).squeeze(0)
        uncertainty = torch.sigmoid(self.uncertainty(pooled)).squeeze()
        return context, uncertainty


class RFConditionBranch(nn.Module):
    """Small optional IQ quality branch from the design report."""

    def __init__(self, q_dim: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, 2, kernel_size=7, padding=3, groups=2),
            nn.Conv1d(2, 32, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(32, 32, kernel_size=5, padding=2, groups=32),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, 64),
            nn.GELU(),
            nn.Linear(64, int(q_dim)),
        )

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(iq, dtype=torch.float32)
        if x.ndim != 3 or x.size(1) != 2:
            raise ValueError(f"expected IQ tensor [N,2,L], got shape={tuple(x.shape)}")
        return self.net(x)


class LowRankFeatureCalibrator(nn.Module):
    """Context-driven low-rank residual feature calibrator."""

    def __init__(self, config: OPGACConfig) -> None:
        super().__init__()
        self.config = config
        dim = int(config.feature_dim)
        rank = int(config.low_rank)
        hidden = int(config.hidden_dim)
        out_dim = 2 * dim * rank + dim + 1
        self.hyper = nn.Sequential(nn.Linear(int(config.context_dim), hidden), nn.GELU(), nn.Linear(hidden, out_dim))
        self.quality_gate = nn.Sequential(
            nn.Linear(dim + int(config.context_dim) + int(config.quality_dim), hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.hyper[-1].weight)
        nn.init.zeros_(self.hyper[-1].bias)

    def forward(
        self,
        features: torch.Tensor,
        context: torch.Tensor,
        *,
        quality_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = _as_feature_matrix(features, self.config.feature_dim, normalize=self.config.normalize_features)
        ctx = torch.as_tensor(context, dtype=x.dtype, device=x.device).view(1, -1)
        if ctx.size(1) != self.config.context_dim:
            raise ValueError(f"context must have dim {self.config.context_dim}")
        params = self.hyper(ctx).squeeze(0)
        dim = int(self.config.feature_dim)
        rank = int(self.config.low_rank)
        uv_len = dim * rank
        u = params[:uv_len].view(dim, rank)
        v = params[uv_len : 2 * uv_len].view(dim, rank)
        bias = params[2 * uv_len : 2 * uv_len + dim].view(1, dim)
        alpha_raw = params[-1]
        alpha = torch.sigmoid(alpha_raw) * float(self.config.max_alpha)
        if quality_features is None:
            q = torch.zeros(x.size(0), self.config.quality_dim, dtype=x.dtype, device=x.device)
        else:
            q = torch.as_tensor(quality_features, dtype=x.dtype, device=x.device)
        if q.ndim != 2 or q.size(0) != x.size(0) or q.size(1) != self.config.quality_dim:
            raise ValueError("quality_features must be [N, quality_dim]")
        ctx_rows = ctx.expand(x.size(0), -1)
        gate = torch.sigmoid(self.quality_gate(torch.cat([x, q, ctx_rows], dim=1)))
        low_rank_delta = (x @ v) @ u.T
        calibrated = x + gate * alpha * low_rank_delta + gate * alpha * bias
        if self.config.normalize_features:
            calibrated = normalize_rows(calibrated)
        else:
            calibrated = F.layer_norm(calibrated, (dim,))
        return calibrated, {"alpha": alpha.detach(), "quality_gate": gate.detach()}


class OldGaussianMemoryCalibrator(nn.Module):
    """Calibrate old-class Gaussian states using target-old support only."""

    def __init__(self, config: OPGACConfig) -> None:
        super().__init__()
        self.config = config
        hidden = int(config.hidden_dim)
        inp = int(config.context_dim + config.feature_dim + 4)
        self.mean_delta = nn.Sequential(nn.Linear(inp, hidden), nn.GELU(), nn.Linear(hidden, int(config.feature_dim)))
        self.var_scale = nn.Sequential(nn.Linear(inp, hidden), nn.GELU(), nn.Linear(hidden, int(config.feature_dim)))
        self.threshold_delta = nn.Sequential(nn.Linear(inp, hidden), nn.GELU(), nn.Linear(hidden, 1))
        for block in (self.mean_delta, self.var_scale, self.threshold_delta):
            nn.init.zeros_(block[-1].weight)
            nn.init.zeros_(block[-1].bias)

    def forward(
        self,
        memory: OPGACMemory,
        support_features: torch.Tensor,
        support_labels: torch.Tensor | Iterable[int],
        context: torch.Tensor,
    ) -> OPGACMemory:
        if not memory.old_states:
            return memory.clone()
        device = next(iter(memory.old_states.values())).means.device
        support = _as_feature_matrix(support_features, self.config.feature_dim, normalize=self.config.normalize_features).to(device)
        labels = _as_labels(support_labels, device=support.device)
        ctx = torch.as_tensor(context, dtype=support.dtype, device=support.device).view(1, -1)
        updated = memory.clone()
        for label, base_state in memory.old_states.items():
            state = base_state.clone()
            mask = labels == int(label)
            n = int(mask.sum().item())
            if n <= 0:
                updated.old_states[int(label)] = state
                continue
            members = support[mask]
            state.means = state.means.clone()
            state.diag_vars = state.diag_vars.clone()
            state.component_thresholds = state.component_thresholds.clone()
            assigned = self._assign_members_to_components(members, state)
            component_updates = []
            for comp_idx in range(state.means.size(0)):
                comp_members = members[assigned == int(comp_idx)]
                comp_n = int(comp_members.size(0))
                if comp_n <= 0:
                    continue
                target_mean = comp_members.mean(dim=0, keepdim=True)
                if self.config.normalize_features:
                    target_mean = normalize_rows(target_mean)
                mean_residual = target_mean.squeeze(0) - state.means[comp_idx]
                base_rho = comp_n / (comp_n + max(0.0, float(self.config.old_shrinkage_kappa)))
                compact = (comp_members - target_mean).pow(2).mean().sqrt()
                response = torch.tensor(
                    [float(comp_n), float(base_rho), float(compact.item()), float(state.support_count)],
                    dtype=support.dtype,
                    device=support.device,
                ).view(1, -1)
                cond = torch.cat([ctx, state.means[comp_idx].view(1, -1), response], dim=1)
                learned_delta = self.mean_delta(cond).squeeze(0)
                delta = base_rho * mean_residual + base_rho * learned_delta
                state.means[comp_idx] = state.means[comp_idx] + delta
                if comp_n > 1:
                    sample_var = (comp_members - state.means[comp_idx].view(1, -1)).pow(2).mean(dim=0).clamp_min(
                        self.config.min_variance
                    )
                else:
                    sample_var = state.diag_vars[comp_idx]
                learned_scale = torch.exp(self.var_scale(cond).squeeze(0)).clamp(
                    self.config.min_variance_scale,
                    self.config.max_variance_scale,
                )
                state.diag_vars[comp_idx] = (
                    (1.0 - base_rho) * state.diag_vars[comp_idx] + base_rho * sample_var
                ).clamp_min(self.config.min_variance) * learned_scale
                support_d2 = (
                    (comp_members - state.means[comp_idx].view(1, -1)).pow(2)
                    / state.diag_vars[comp_idx].view(1, -1)
                ).sum(dim=1)
                q = torch.quantile(support_d2, min(max(float(self.config.default_component_threshold_quantile), 0.5), 0.999))
                learned_tau = torch.exp(self.threshold_delta(cond).squeeze()).clamp(0.5, 2.0)
                tau = torch.maximum(state.component_thresholds[comp_idx], q) * learned_tau
                state.component_thresholds[comp_idx] = tau.clamp(self.config.min_threshold, self.config.max_threshold)
                component_updates.append(
                    {
                        "component": int(comp_idx),
                        "support_count": int(comp_n),
                        "base_rho": float(base_rho),
                        "mean_shift_norm": float(delta.norm().item()),
                        "compactness": float(compact.item()),
                    }
                )
            if self.config.normalize_features:
                state.means = normalize_rows(state.means)
            state.class_threshold = float(max(float(state.class_threshold), float(state.component_thresholds.max().item())))
            state.support_count = int(state.support_count + n)
            state.version += 1
            state.metadata = {
                **state.metadata,
                "target_old_support_count": n,
                "component_updates": component_updates,
            }
            updated.old_states[int(label)] = state
        updated.version += 1
        updated.update_log.append({"type": "old_memory_calibration", "support_count": int(labels.numel()), "version": updated.version})
        return updated

    def _assign_members_to_components(self, members: torch.Tensor, state: GaussianClassState) -> torch.Tensor:
        means = state.means.to(members.device, members.dtype)
        diag = state.diag_vars.to(members.device, members.dtype).clamp_min(self.config.min_variance)
        d2 = ((members[:, None, :] - means[None, :, :]).pow(2) / diag[None, :, :]).sum(dim=-1)
        return torch.argmin(d2, dim=1)


class NewClassGaussianGenerator(nn.Module):
    """Few-shot seen-new Gaussian generator with old covariance priors."""

    def __init__(self, config: OPGACConfig) -> None:
        super().__init__()
        self.config = config
        hidden = int(config.hidden_dim)
        inp = int(config.context_dim + 2 * config.feature_dim + 3)
        self.mean_correction = nn.Sequential(nn.Linear(inp, hidden), nn.GELU(), nn.Linear(hidden, int(config.feature_dim)))
        nn.init.zeros_(self.mean_correction[-1].weight)
        nn.init.zeros_(self.mean_correction[-1].bias)

    def _nearest_old_prior(
        self,
        support_mean: torch.Tensor,
        memory: OPGACMemory,
        k: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        means = []
        vars_ = []
        thresholds = []
        for state in memory.old_states.values():
            means.append(state.means)
            vars_.append(state.diag_vars)
            thresholds.append(state.component_thresholds)
        if not means:
            raise ValueError("new-class registration requires at least one old state prior")
        all_means = torch.cat(means, dim=0).to(support_mean.device)
        all_vars = torch.cat(vars_, dim=0).to(support_mean.device)
        all_thresholds = torch.cat(thresholds, dim=0).to(support_mean.device)
        d2 = (all_means - support_mean.view(1, -1)).pow(2).sum(dim=1)
        top_k = min(max(1, int(k)), int(d2.numel()))
        vals, idx = torch.topk(-d2, top_k)
        weights = torch.softmax(vals, dim=0)
        prior_mean = (weights.view(-1, 1) * all_means[idx]).sum(dim=0)
        prior_var = (weights.view(-1, 1) * all_vars[idx]).sum(dim=0).clamp_min(self.config.min_variance)
        prior_threshold = (weights * all_thresholds[idx]).sum()
        return prior_mean, prior_var, prior_threshold

    def register(
        self,
        memory: OPGACMemory,
        target_new_support: torch.Tensor,
        target_new_labels: torch.Tensor | Iterable[int],
        context: torch.Tensor,
    ) -> OPGACMemory:
        support = _as_feature_matrix(target_new_support, self.config.feature_dim, normalize=self.config.normalize_features)
        labels = _as_labels(target_new_labels, device=support.device)
        if any(int(v) in memory.old_states for v in labels.detach().cpu().tolist()):
            overlap = sorted({int(v) for v in labels.detach().cpu().tolist()} & set(memory.old_states))
            raise ValueError(f"new-class labels overlap old-class memory: {overlap}")
        ctx = torch.as_tensor(context, dtype=support.dtype, device=support.device).view(1, -1)
        updated = memory.clone()
        for label in sorted({int(v) for v in labels.detach().cpu().tolist() if int(v) != UNKNOWN_LABEL}):
            members = support[labels == int(label)]
            n = int(members.size(0))
            if n <= 0:
                continue
            sample_mean = members.mean(dim=0)
            if self.config.normalize_features:
                sample_mean = normalize_rows(sample_mean.view(1, -1)).squeeze(0)
            prior_mean, prior_var, prior_tau = self._nearest_old_prior(sample_mean, updated, self.config.new_mean_prior_k)
            rho = n / (n + max(0.0, float(self.config.new_shrinkage_nu)))
            cond = torch.cat(
                [
                    ctx,
                    sample_mean.view(1, -1),
                    prior_mean.view(1, -1),
                    torch.tensor([[float(n), float(rho), float(prior_tau.item())]], dtype=support.dtype, device=support.device),
                ],
                dim=1,
            )
            correction = self.mean_correction(cond).squeeze(0)
            mean = sample_mean + rho * correction
            if self.config.normalize_features:
                mean = normalize_rows(mean.view(1, -1)).squeeze(0)
            if n > 1:
                sample_var = (members - mean.view(1, -1)).pow(2).mean(dim=0).clamp_min(self.config.min_variance)
            else:
                sample_var = prior_var
            cov_rho = n / (n + max(0.0, float(self.config.cov_shrinkage_nu)))
            diag_var = ((1.0 - cov_rho) * prior_var + cov_rho * sample_var).clamp_min(self.config.min_variance)
            d2 = ((members - mean.view(1, -1)).pow(2) / diag_var.view(1, -1)).sum(dim=1)
            sample_tau = d2.max() if d2.numel() else prior_tau
            tau_rho = n / (n + max(0.0, float(self.config.threshold_shrinkage_nu)))
            tau = ((1.0 - tau_rho) * prior_tau + tau_rho * sample_tau).clamp(
                self.config.min_threshold,
                self.config.max_threshold,
            )
            overlap_state, overlap_d2, overlap_tau = self._nearest_old_overlap(mean, diag_var, updated)
            lifecycle = "confirmed"
            if overlap_state is not None and float(overlap_d2) <= float(overlap_tau) + float(self.config.overlap_margin):
                lifecycle = "provisional"
            state = _state_from_single_component(
                class_id=int(label),
                group="seen_new",
                mean=mean,
                diag_var=diag_var,
                threshold=float(tau.item()),
                class_threshold=float(max(float(tau.item()), self.config.default_class_threshold)),
                count=n,
                metadata={
                    "source": "target_new_support",
                    "prior": "nearest_old_covariance",
                    "nearest_old_overlap_class": None if overlap_state is None else int(overlap_state.class_id),
                    "nearest_old_overlap_d2": None if overlap_state is None else float(overlap_d2),
                    "nearest_old_overlap_tau": None if overlap_state is None else float(overlap_tau),
                },
            )
            state.lifecycle = lifecycle
            updated.new_states[int(label)] = state
        updated.version += 1
        updated.update_log.append({"type": "new_class_registration", "support_count": int(labels.numel()), "version": updated.version})
        return updated

    def _nearest_old_overlap(
        self,
        mean: torch.Tensor,
        diag_var: torch.Tensor,
        memory: OPGACMemory,
    ) -> tuple[GaussianClassState | None, float, float]:
        best_state = None
        best_d2 = float("inf")
        best_tau = float("inf")
        for state in memory.old_states.values():
            d2 = ((state.means - mean.view(1, -1)).pow(2) / diag_var.view(1, -1).clamp_min(self.config.min_variance)).sum(dim=1)
            idx = int(torch.argmin(d2).item())
            if float(d2[idx].item()) < best_d2:
                best_state = state
                best_d2 = float(d2[idx].item())
                best_tau = float(state.component_thresholds[idx].item())
        return best_state, best_d2, best_tau


class EnergyRejectionHead(nn.Module):
    """Old/new/unknown/ambiguous GMM energy decision head."""

    def __init__(self, config: OPGACConfig) -> None:
        super().__init__()
        self.config = config

    def _score_group(
        self,
        features: torch.Tensor,
        states: Mapping[int, GaussianClassState],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        n = int(features.size(0))
        if not states:
            nan = torch.full((n,), float("nan"), dtype=features.dtype, device=features.device)
            labels = torch.full((n,), UNKNOWN_LABEL, dtype=torch.long, device=features.device)
            return nan, labels, nan, nan, {}
        rows = []
        class_labels = []
        component_d2_cols = []
        component_gate_cols = []
        for label, state in sorted(states.items()):
            nll, d2 = _component_nll(features, state, self.config.eps)
            norm_energy = (nll - float(state.energy_median)) / max(float(state.energy_iqr), self.config.eps)
            rows.append(norm_energy)
            class_labels.append(int(label))
            min_d2, min_idx = d2.min(dim=1)
            component_d2_cols.append(min_d2)
            thresholds = state.component_thresholds.to(features.device, features.dtype)[min_idx]
            component_gate_cols.append(min_d2 <= thresholds)
        energy_matrix = torch.stack(rows, dim=1)
        best_idx = torch.argmin(energy_matrix, dim=1)
        best_energy = energy_matrix.gather(1, best_idx.view(-1, 1)).squeeze(1)
        labels = torch.tensor(class_labels, dtype=torch.long, device=features.device)[best_idx]
        d2_matrix = torch.stack(component_d2_cols, dim=1)
        d2_best = d2_matrix.gather(1, best_idx.view(-1, 1)).squeeze(1)
        gate_matrix = torch.stack(component_gate_cols, dim=1)
        component_pass = gate_matrix.gather(1, best_idx.view(-1, 1)).squeeze(1)
        if energy_matrix.size(1) > 1:
            top2 = torch.topk(-energy_matrix, 2, dim=1).values
            margin = top2[:, 0] - top2[:, 1]
        else:
            margin = torch.full_like(best_energy, float("inf"))
        diagnostics = {
            "energy_matrix": energy_matrix.detach(),
            "component_gate_pass": component_pass.detach(),
        }
        return best_energy, labels, d2_best, margin, diagnostics

    def forward(self, features: torch.Tensor, memory: OPGACMemory) -> OPGACPrediction:
        x = _as_feature_matrix(features, self.config.feature_dim, normalize=self.config.normalize_features)
        old_e, old_label, old_d2, old_margin, old_diag = self._score_group(x, memory.old_states)
        new_e, new_label, new_d2, new_margin, new_diag = self._score_group(x, memory.new_states)
        decisions: list[str] = []
        predicted = torch.full((x.size(0),), UNKNOWN_LABEL, dtype=torch.long, device=x.device)
        accepted = torch.zeros((x.size(0),), dtype=torch.bool, device=x.device)
        reasons: list[list[str]] = []
        all_states = memory.all_states()
        for row in range(x.size(0)):
            row_reasons: list[str] = []
            old_state = memory.old_states.get(int(old_label[row].item()))
            new_state = memory.new_states.get(int(new_label[row].item()))
            old_gate = False
            new_gate = False
            if old_state is not None:
                old_gate = (
                    torch.isfinite(old_e[row])
                    and float(old_e[row].item()) <= float(old_state.class_threshold)
                    and float(old_margin[row].item()) >= float(self.config.top2_margin)
                )
                if not old_gate:
                    row_reasons.append("OLD_LOW_LIKELIHOOD")
            if new_state is not None and new_state.lifecycle == "confirmed":
                new_gate = (
                    torch.isfinite(new_e[row])
                    and float(new_e[row].item()) <= float(new_state.class_threshold)
                    and float(new_margin[row].item()) >= float(self.config.top2_margin)
                )
                if not new_gate:
                    row_reasons.append("NEW_LOW_LIKELIHOOD")
            elif new_state is not None and new_state.lifecycle != "confirmed":
                row_reasons.append("NEW_PROVISIONAL")
            if old_state is None and new_state is None:
                decisions.append(DECISION_UNKNOWN)
                row_reasons.append("NO_REGISTERED_CLASS")
            elif old_gate and (not new_gate or float(old_e[row].item()) + self.config.old_new_margin < float(new_e[row].item())):
                decisions.append(DECISION_OLD)
                predicted[row] = int(old_label[row].item())
                accepted[row] = True
            elif new_gate and (not old_gate or float(new_e[row].item()) + self.config.old_new_margin < float(old_e[row].item())):
                decisions.append(DECISION_NEW)
                predicted[row] = int(new_label[row].item())
                accepted[row] = True
            elif not old_gate and not new_gate:
                decisions.append(DECISION_UNKNOWN)
                row_reasons.append("LOW_LIKELIHOOD")
            else:
                decisions.append(DECISION_AMBIGUOUS)
                row_reasons.append("OLD_NEW_AMBIGUOUS")
            if predicted[row].item() != UNKNOWN_LABEL:
                state = all_states[int(predicted[row].item())]
                best_d2 = old_d2[row] if state.group == "old" else new_d2[row]
                if float(best_d2.item()) > float(state.component_thresholds.max().item()):
                    decisions[-1] = DECISION_AMBIGUOUS if self.config.ambiguous_on_overlap else DECISION_UNKNOWN
                    accepted[row] = False
                    predicted[row] = UNKNOWN_LABEL
                    row_reasons.append("OUTSIDE_GAUSSIAN_ELLIPSOID")
            reasons.append(row_reasons or ["ACCEPTED"])
        best_d2 = torch.minimum(
            torch.nan_to_num(old_d2, nan=float("inf")),
            torch.nan_to_num(new_d2, nan=float("inf")),
        )
        diagnostics = {
            "old_energy_matrix": old_diag.get("energy_matrix", torch.empty(0)).detach().cpu(),
            "new_energy_matrix": new_diag.get("energy_matrix", torch.empty(0)).detach().cpu(),
            "old_component_gate_pass": old_diag.get("component_gate_pass", torch.empty(0)).detach().cpu(),
            "new_component_gate_pass": new_diag.get("component_gate_pass", torch.empty(0)).detach().cpu(),
        }
        return OPGACPrediction(
            decisions=decisions,
            predicted_labels=predicted.detach().cpu(),
            accepted=accepted.detach().cpu(),
            best_old_labels=old_label.detach().cpu(),
            best_new_labels=new_label.detach().cpu(),
            old_scores=old_e.detach().cpu(),
            new_scores=new_e.detach().cpu(),
            best_component_d2=best_d2.detach().cpu(),
            margin_old_new=(new_e - old_e).detach().cpu(),
            margin_top2=torch.minimum(torch.nan_to_num(old_margin, nan=float("inf")), torch.nan_to_num(new_margin, nan=float("inf"))).detach().cpu(),
            reject_reasons=reasons,
            diagnostics=diagnostics,
        )


class OPGACNet(nn.Module):
    """Support-conditioned OPGAC-Net orchestrator.

    `initialize_memory` is the only adaptation path. `predict` does not update
    context, thresholds, prototypes, or feature-calibrator parameters.
    """

    def __init__(self, config: OPGACConfig) -> None:
        super().__init__()
        self.config = config
        self.context_encoder = DeepSetContextEncoder(config)
        self.feature_calibrator = LowRankFeatureCalibrator(config)
        self.old_memory_calibrator = OldGaussianMemoryCalibrator(config)
        self.new_class_generator = NewClassGaussianGenerator(config)
        self.rejection_head = EnergyRejectionHead(config)

    def initialize_memory(
        self,
        ground_memory: OPGACMemory,
        *,
        stage: str,
        target_old_support: torch.Tensor | None = None,
        target_old_labels: torch.Tensor | Iterable[int] | None = None,
        target_new_support: torch.Tensor | None = None,
        target_new_labels: torch.Tensor | Iterable[int] | None = None,
        support_quality: torch.Tensor | None = None,
        support_stats: torch.Tensor | None = None,
    ) -> OPGACMemory:
        use_old = target_old_support is not None and target_old_labels is not None and torch.as_tensor(target_old_support).numel() > 0
        use_new = target_new_support is not None and target_new_labels is not None and torch.as_tensor(target_new_support).numel() > 0
        validate_stage2_protocol(
            stage,
            use_target_old_support=bool(use_old),
            use_target_new_support=bool(use_new),
            use_unknown_query_for_threshold_calibration=False,
        )
        support_parts = []
        label_parts = []
        if use_old:
            old_support = _as_feature_matrix(target_old_support, self.config.feature_dim, normalize=self.config.normalize_features)
            old_labels = _as_labels(target_old_labels, device=old_support.device)
            unknown_in_old = [int(v) for v in old_labels.detach().cpu().tolist() if int(v) == UNKNOWN_LABEL]
            if unknown_in_old:
                raise ValueError("target_old_support cannot contain unknown labels")
            support_parts.append(old_support)
            label_parts.append(old_labels)
        if use_new:
            new_support = _as_feature_matrix(target_new_support, self.config.feature_dim, normalize=self.config.normalize_features)
            new_labels = _as_labels(target_new_labels, device=new_support.device)
            if any(int(v) == UNKNOWN_LABEL for v in new_labels.detach().cpu().tolist()):
                raise ValueError("target_new_support cannot contain unknown labels")
            support_parts.append(new_support)
            label_parts.append(new_labels)
        if support_parts:
            support_all = torch.cat(support_parts, dim=0)
            if support_stats is not None:
                stats = torch.as_tensor(support_stats, dtype=support_all.dtype, device=support_all.device)
                if stats.size(0) != support_all.size(0):
                    raise ValueError("support_stats must align with old+new support rows")
            else:
                stats = self._support_stats_against_memory(support_all, ground_memory)
            if support_quality is not None:
                quality = torch.as_tensor(support_quality, dtype=support_all.dtype, device=support_all.device)
                if quality.size(0) != support_all.size(0):
                    raise ValueError("support_quality must align with old+new support rows")
            else:
                quality = None
            context, uncertainty = self.context_encoder(support_all, quality_features=quality, support_stats=stats)
        else:
            context = torch.zeros(self.config.context_dim, dtype=torch.float32)
            uncertainty = torch.tensor(1.0, dtype=torch.float32)
        memory = ground_memory.clone()
        memory.domain_context = context.detach().clone()
        memory.uncertainty = float(uncertainty.detach().item())
        if use_old:
            old_calibrated, _ = self.feature_calibrator(old_support, context)
            memory = self.old_memory_calibrator(memory, old_calibrated, old_labels, context)
            memory.domain_context = context.detach().clone()
            memory.uncertainty = float(uncertainty.detach().item())
        if use_new:
            new_calibrated, _ = self.feature_calibrator(new_support, context)
            memory = self.new_class_generator.register(memory, new_calibrated, new_labels, context)
            memory.domain_context = context.detach().clone()
            memory.uncertainty = float(uncertainty.detach().item())
        return memory

    def predict(
        self,
        features: torch.Tensor,
        memory: OPGACMemory,
        *,
        quality_features: torch.Tensor | None = None,
    ) -> OPGACPrediction:
        x = _as_feature_matrix(features, self.config.feature_dim, normalize=self.config.normalize_features)
        context = memory.domain_context
        if context is None:
            context = torch.zeros(self.config.context_dim, dtype=x.dtype, device=x.device)
        calibrated, telemetry = self.feature_calibrator(x, context.to(x.device), quality_features=quality_features)
        prediction = self.rejection_head(calibrated, memory)
        prediction.diagnostics["feature_calibrator_alpha"] = telemetry["alpha"].detach().cpu().view(1)
        prediction.diagnostics["feature_quality_gate"] = telemetry["quality_gate"].detach().cpu().view(-1)
        return prediction

    def _support_stats_against_memory(self, support: torch.Tensor, memory: OPGACMemory) -> torch.Tensor:
        states = memory.all_states()
        if not states:
            return torch.zeros(support.size(0), self.config.stats_dim, dtype=support.dtype, device=support.device)
        energy_cols = []
        d2_cols = []
        for state in states.values():
            nll, d2 = _component_nll(support, state, self.config.eps)
            energy_cols.append(nll)
            d2_cols.append(d2.min(dim=1).values)
        e = torch.stack(energy_cols, dim=1)
        d2 = torch.stack(d2_cols, dim=1)
        best_e, _ = e.min(dim=1)
        best_d2, _ = d2.min(dim=1)
        if e.size(1) > 1:
            top2_e = torch.topk(-e, 2, dim=1).values
            margin_e = top2_e[:, 0] - top2_e[:, 1]
        else:
            margin_e = torch.ones_like(best_e)
        high_conf = ((best_e <= self.config.default_class_threshold) & (best_d2 <= self.config.default_class_threshold)).float()
        rejected = 1.0 - high_conf
        rows = torch.stack([best_d2, best_e, margin_e, high_conf, rejected, torch.ones_like(best_e)], dim=1)
        if self.config.stats_dim == rows.size(1):
            return rows
        if self.config.stats_dim < rows.size(1):
            return rows[:, : self.config.stats_dim]
        pad = torch.zeros(rows.size(0), self.config.stats_dim - rows.size(1), dtype=rows.dtype, device=rows.device)
        return torch.cat([rows, pad], dim=1)


def rollback_memory(memory: OPGACMemory, target_version: int | None = None) -> OPGACMemory:
    """Rollback mutable target memory to the immutable ground old bank.

    The first implementation intentionally rolls back all target-side new class
    registrations and old-class calibrations. Fine-grained checkpoint replay can
    be layered on top once remote experiments need it.
    """

    restored = OPGACMemory(old_states={int(k): v.clone() for k, v in memory.ground_old_states.items()})
    restored.version = int(target_version or (memory.version + 1))
    restored.update_log = [dict(row) for row in memory.update_log]
    restored.update_log.append({"type": "rollback_to_ground_old_memory", "version": restored.version})
    return restored


def drift_alarm(memory: OPGACMemory, *, config: OPGACConfig) -> dict:
    """Return simple memory-health alarms for on-orbit update supervision."""

    alarms: list[str] = []
    shifts = {}
    for label, state in memory.old_states.items():
        base = memory.ground_old_states.get(int(label))
        if base is None:
            continue
        shift = torch.cdist(state.means[:1], base.means[:1]).item()
        shifts[int(label)] = float(shift)
        if shift >= float(config.drift_alarm_shift):
            alarms.append(f"old_class_{label}_mean_shift")
    for label, state in memory.new_states.items():
        if state.lifecycle != "confirmed":
            alarms.append(f"new_class_{label}_{state.lifecycle}")
    return {"alarms": alarms, "old_mean_shifts": shifts, "version": int(memory.version)}


def register_old_classes_opgac(
    ground_memory: OPGACMemory,
    target_old_support: torch.Tensor,
    target_old_labels: torch.Tensor | Iterable[int],
    *,
    config: OPGACConfig,
    model: OPGACNet | None = None,
    stage: str = "Stage2-B",
) -> tuple[OPGACMemory, OPGACNet]:
    """Thin Stage2-B/C old-class OPGAC registration entrypoint."""

    net = model or OPGACNet(config)
    memory = net.initialize_memory(
        ground_memory,
        stage=stage,
        target_old_support=target_old_support,
        target_old_labels=target_old_labels,
    )
    return memory, net


def register_new_classes_opgac(
    memory: OPGACMemory,
    target_new_support: torch.Tensor,
    target_new_labels: torch.Tensor | Iterable[int],
    *,
    config: OPGACConfig,
    model: OPGACNet | None = None,
    stage: str = "Stage2-C",
) -> tuple[OPGACMemory, OPGACNet]:
    """Thin Stage2-C seen-new OPGAC registration entrypoint."""

    net = model or OPGACNet(config)
    updated = net.initialize_memory(
        memory,
        stage=stage,
        target_new_support=target_new_support,
        target_new_labels=target_new_labels,
    )
    return updated, net


def opgac_to_prediction_result(result: OPGACPrediction) -> PredictionResult:
    """Map OPGAC decisions into the existing feature-eval result contract."""

    score = -torch.minimum(
        torch.nan_to_num(result.old_scores.float(), nan=float("inf")),
        torch.nan_to_num(result.new_scores.float(), nan=float("inf")),
    )
    decisions = []
    for decision in result.decisions:
        if decision == DECISION_UNKNOWN:
            decisions.append("reject")
        elif decision == DECISION_AMBIGUOUS:
            decisions.append("uncertain")
        else:
            decisions.append("accept")
    diagnostics = {
        **{key: value for key, value in result.diagnostics.items()},
        "opgac_old_score": result.old_scores.float(),
        "opgac_new_score": result.new_scores.float(),
        "opgac_old_new_margin": result.margin_old_new.float(),
        "opgac_top2_margin": result.margin_top2.float(),
        "opgac_best_component_d2": result.best_component_d2.float(),
        "opgac_best_old_label": result.best_old_labels.long(),
        "opgac_best_new_label": result.best_new_labels.long(),
    }
    return PredictionResult(
        predicted_labels=result.predicted_labels.long(),
        scores=score.float(),
        accepted=result.accepted.bool(),
        candidate_labels=result.predicted_labels.long(),
        diagnostics=diagnostics,
        margins=result.margin_top2.float(),
        mahalanobis=result.best_component_d2.float(),
        gate_reasons=[";".join(row) for row in result.reject_reasons],
        decisions=decisions,
        energy=torch.minimum(
            torch.nan_to_num(result.old_scores.float(), nan=float("inf")),
            torch.nan_to_num(result.new_scores.float(), nan=float("inf")),
        ),
    )


def predict_with_opgac_head(
    features: torch.Tensor,
    memory: OPGACMemory,
    *,
    config: OPGACConfig,
    model: OPGACNet | None = None,
    quality_features: torch.Tensor | None = None,
) -> PredictionResult:
    """Predict with fixed OPGAC memory and return existing PredictionResult."""

    net = model or OPGACNet(config)
    return opgac_to_prediction_result(net.predict(features, memory, quality_features=quality_features))


__all__ = [
    "DECISION_AMBIGUOUS",
    "DECISION_NEW",
    "DECISION_OLD",
    "DECISION_UNKNOWN",
    "DeepSetContextEncoder",
    "EnergyRejectionHead",
    "FixedFeatureTransform",
    "GaussianClassState",
    "LowRankFeatureCalibrator",
    "NewClassGaussianGenerator",
    "OldGaussianMemoryCalibrator",
    "OPGACConfig",
    "OPGACMemory",
    "OPGACNet",
    "OPGACPrediction",
    "RFConditionBranch",
    "build_old_memory_from_prototypes",
    "drift_alarm",
    "fit_fixed_feature_transform",
    "opgac_to_prediction_result",
    "predict_with_opgac_head",
    "register_new_classes_opgac",
    "register_old_classes_opgac",
    "rollback_memory",
]
