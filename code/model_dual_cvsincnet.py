
from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

_CODE_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (_CODE_ROOT, _REPO_ROOT):
    _text = str(_path)
    if _text in sys.path:
        sys.path.remove(_text)
for _path in (_REPO_ROOT, _CODE_ROOT):
    sys.path.insert(0, str(_path))

try:
    from model_modified import build_model as build_single_model
except Exception:
    from model import build_model as build_single_model

try:
    from baselines.common.resnet1d import MLPClassifier, ResNet1DEncoder
    from baselines.cvcnn_ce.model import BasicCVCNN, SincCVCNN
except Exception:  # pragma: no cover - import errors surface when a baseline family is requested.
    MLPClassifier = None
    ResNet1DEncoder = None
    BasicCVCNN = None
    SincCVCNN = None


def build_single_model_compat(**kwargs):
    try:
        return build_single_model(**kwargs)
    except TypeError:
        fallback = dict(kwargs)
        for key in (
            "branch_ablation",
            "model_variant",
            "mixstyle_on",
            "mixstyle_p",
            "mixstyle_alpha",
            "mixstyle_eps",
            "mixstyle_layers",
            "mixstyle_use_domain_label",
            "mixstyle_mix",
            "mixstyle_strength",
            "mixstyle_fallback",
            "time_stability_mode",
            "freq_stability_mode",
            "time_stability_channels",
            "freq_stability_channels",
            "use_crra",
            "crra_rank",
            "crra_alpha_max",
            "crra_shrinkage",
            "crra_condition_dim",
            "crra_nuisance_dim",
            "crra_start_epoch",
            "crra_ramp_epochs",
        ):
            fallback.pop(key, None)
        return build_single_model(**fallback)


def backbone_forward_compat(
    backbone,
    x,
    *,
    y=None,
    return_aux: bool = True,
    domain_labels=None,
    crra_epoch: Optional[int] = None,
    update_crra_support: bool = False,
    crra_support_mask: Optional[torch.Tensor] = None,
):
    try:
        return backbone(
            x,
            y=y,
            return_aux=return_aux,
            domain_labels=domain_labels,
            crra_epoch=crra_epoch,
            update_crra_support=update_crra_support,
            crra_support_mask=crra_support_mask,
        )
    except TypeError:
        return backbone(x, y=y, return_aux=return_aux)


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return GradReverse.apply(x, lambd)


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: Optional[int] = None, drop: float = 0.1):
        super().__init__()
        if hidden is None:
            hidden = max(64, in_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def iq_to_complex(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 3 or int(x.size(1)) != 2:
        raise ValueError("IQ input must have shape [batch, 2, time]")
    return torch.complex(x[:, 0].float(), x[:, 1].float())


def complex_to_iq(z: torch.Tensor, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    if not torch.is_complex(z) or z.ndim != 2:
        raise ValueError("complex waveform must have shape [batch, time]")
    out = torch.stack([z.real, z.imag], dim=1)
    return out.to(dtype=dtype or out.dtype)


class NuisanceEstimator(nn.Module):
    """Predict only normalized CFO, common phase and scalar log-gain."""

    def __init__(
        self,
        hidden: int = 8,
        max_cfo_cycles_per_sample: float = 0.05,
        max_log_gain: float = 2.0,
    ):
        super().__init__()
        self.max_cfo_cycles_per_sample = float(max_cfo_cycles_per_sample)
        self.max_log_gain = float(max_log_gain)
        self.features = nn.Sequential(
            nn.Conv1d(2, int(hidden), kernel_size=7, padding=3, bias=False),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(int(hidden), 3)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.head(self.features(x.float()).squeeze(-1))
        return torch.stack(
            [
                torch.tanh(raw[:, 0]) * self.max_cfo_cycles_per_sample,
                torch.tanh(raw[:, 1]) * math.pi,
                torch.tanh(raw[:, 2]) * self.max_log_gain,
            ],
            dim=1,
        )


class AnalyticCanonicalizer(nn.Module):
    """Analytic inverse for CFO, common phase and scalar gain only."""

    def forward(self, x: torch.Tensor, nuisance_coef: torch.Tensor) -> torch.Tensor:
        if nuisance_coef.ndim != 2 or tuple(nuisance_coef.shape[1:]) != (3,):
            raise ValueError("nuisance_coef must have shape [batch, 3]")
        z = iq_to_complex(x)
        coef = nuisance_coef.to(device=z.device, dtype=torch.float32)
        n = torch.arange(z.size(-1), device=z.device, dtype=torch.float32).unsqueeze(0)
        phase = 2.0 * math.pi * coef[:, 0:1] * n + coef[:, 1:2]
        canonical = z * torch.exp(torch.complex(torch.zeros_like(phase), -phase))
        canonical = canonical * torch.exp(-coef[:, 2:3])
        return complex_to_iq(canonical, dtype=x.dtype)


class ContentEstimator(nn.Module):
    """Bandwidth-limited blind content estimator with per-sample confidence."""

    def __init__(self, kernel_size: int = 5):
        super().__init__()
        kernel_size = int(kernel_size)
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer >= 3")
        self.smoother = nn.Conv1d(
            2,
            2,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=2,
            bias=False,
        )
        nn.init.constant_(self.smoother.weight, 1.0 / float(kernel_size))
        self.residual_mix_logit = nn.Parameter(torch.tensor(0.0))
        self.confidence_bias = nn.Parameter(torch.tensor(1.5))
        self.confidence_scale_raw = nn.Parameter(torch.tensor(0.0))

    def forward(self, canonical_iq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = canonical_iq.float()
        filtered = self.smoother(x)
        mix = torch.sigmoid(self.residual_mix_logit)
        estimate_iq = mix * x + (1.0 - mix) * filtered
        residual_power = (x - estimate_iq).square().sum(dim=1)
        scale = torch.nn.functional.softplus(self.confidence_scale_raw) + 1e-6
        confidence = torch.sigmoid(self.confidence_bias - scale * residual_power)
        return iq_to_complex(estimate_iq), confidence


class ResponseBasis(nn.Module):
    """Fixed 28-column physical response dictionary used by ECRS V1."""

    block_slices = {
        "pa": slice(0, 8),
        "iq": slice(8, 16),
        "cross": slice(16, 20),
        "slew": slice(20, 28),
    }

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "amplitude_centers",
            torch.tensor([0.15, 0.45, 0.75, 1.05], dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer("amplitude_width", torch.tensor(0.30), persistent=True)

    @staticmethod
    def _delay(value: torch.Tensor, steps: int = 1) -> torch.Tensor:
        steps = int(steps)
        if steps <= 0:
            return value
        return torch.nn.functional.pad(value[..., :-steps], (steps, 0))

    def _rbf(self, amplitude: torch.Tensor) -> torch.Tensor:
        centers = self.amplitude_centers.to(device=amplitude.device, dtype=amplitude.dtype)
        width = self.amplitude_width.to(device=amplitude.device, dtype=amplitude.dtype)
        return torch.exp(-0.5 * ((amplitude.unsqueeze(-1) - centers) / width) ** 2)

    def forward(self, s_hat: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(s_hat) or s_hat.ndim != 2:
            raise ValueError("s_hat must be a complex tensor with shape [batch, time]")
        s_hat = s_hat.to(torch.complex64)
        scale = torch.quantile(s_hat.abs().float(), 0.95, dim=1, keepdim=True).clamp_min(1e-4)
        amplitude = s_hat.abs().float() / scale
        s_prev = self._delay(s_hat, 1)
        amp_prev = self._delay(amplitude, 1)
        b_now = self._rbf(amplitude).to(torch.complex64)
        b_prev = self._rbf(amp_prev).to(torch.complex64)

        pa = torch.cat([s_hat.unsqueeze(-1) * b_now, s_prev.unsqueeze(-1) * b_prev], dim=-1)
        iq = torch.cat(
            [s_hat.conj().unsqueeze(-1) * b_now, s_prev.conj().unsqueeze(-1) * b_prev],
            dim=-1,
        )
        cross = s_hat.unsqueeze(-1) * b_prev
        delta = s_hat - s_prev
        acceleration = delta - self._delay(delta, 1)
        slew = torch.cat(
            [delta.unsqueeze(-1) * b_now, acceleration.unsqueeze(-1) * b_now], dim=-1
        )
        return torch.cat([pa, iq, cross, slew], dim=-1).to(torch.complex64)


class WeightedRidgeLayer(nn.Module):
    """Differentiable complex64 weighted ridge with bounded weights and fallbacks."""

    def __init__(self, alpha_lambda: float = 1e-4, min_weight: float = 0.05, eps: float = 1e-6):
        super().__init__()
        self.alpha_lambda = float(alpha_lambda)
        self.min_weight = float(min_weight)
        self.eps = float(eps)

    @staticmethod
    def _cholesky_solve(matrix: torch.Tensor, rhs: torch.Tensor) -> Optional[torch.Tensor]:
        chol, info = torch.linalg.cholesky_ex(matrix)
        if int(info.item()) != 0:
            return None
        return torch.cholesky_solve(rhs.unsqueeze(-1), chol).squeeze(-1)

    def _coverage(self, amplitude: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        scale = torch.quantile(amplitude, 0.95).clamp_min(self.eps)
        normalized = amplitude / scale
        centers = torch.linspace(0.15, 1.20, 8, device=amplitude.device)
        bins = torch.exp(-0.5 * ((normalized.unsqueeze(-1) - centers) / 0.16) ** 2)
        occupancy = (bins * weight.unsqueeze(-1)).sum(dim=0) / weight.sum().clamp_min(self.eps)
        return (occupancy > 0.02).float().mean()

    def _block_regularization(
        self,
        phi: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor,
        gram_trace_over_k: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_eff = weight.sum().square() / weight.square().sum().clamp_min(self.eps)
        coverage = self._coverage(target.abs().float(), weight)
        total_energy = phi.abs().square().mean().clamp_min(self.eps)
        slew_energy = phi[..., 20:28].abs().square().mean() / total_energy
        q_values = torch.stack(
            [
                coverage,
                coverage,
                (n_eff / float(max(1, phi.size(0)))).clamp(0.05, 1.0),
                slew_energy.float().clamp(0.05, 1.0),
            ]
        ).clamp_min(0.05)
        block_sizes = (8, 8, 4, 8)
        chunks = []
        for q_value, size in zip(q_values, block_sizes):
            value = self.alpha_lambda * gram_trace_over_k / (q_value + self.eps)
            chunks.append(value.expand(size))
        return torch.cat(chunks), coverage

    def forward(
        self,
        phi: torch.Tensor,
        target: torch.Tensor,
        content_confidence: torch.Tensor,
    ) -> Dict[str, object]:
        if phi.ndim != 3 or int(phi.size(-1)) != 28 or not torch.is_complex(phi):
            raise ValueError("phi must be complex [batch, samples, 28]")
        if target.shape != phi.shape[:2] or not torch.is_complex(target):
            raise ValueError("target must be complex [batch, samples]")
        if content_confidence.shape != target.shape:
            raise ValueError("content_confidence must match target")
        coefficients = []
        covariance = []
        infos = []
        eigenvalues = []
        log_conditions = []
        effective_ranks = []
        effective_samples = []
        coverages = []
        nmses = []
        normalized_weights = []

        with torch.autocast(device_type=phi.device.type, enabled=False):
            for batch_index in range(int(phi.size(0))):
                design = phi[batch_index].to(torch.complex64)
                response = target[batch_index].to(torch.complex64)
                weight = content_confidence[batch_index].float().clamp(self.min_weight, 1.0)
                weight = weight / weight.mean().clamp_min(self.eps)
                normalized_weights.append(weight)
                gram = design.conj().transpose(0, 1) @ (weight.unsqueeze(-1) * design)
                rhs = design.conj().transpose(0, 1) @ (weight * response)
                gram_trace_over_k = gram.diagonal().real.mean().clamp_min(self.eps)
                lambda_diag, coverage = self._block_regularization(
                    design, response, weight, gram_trace_over_k
                )
                regularizer = torch.diag(lambda_diag).to(torch.complex64)
                solved = self._cholesky_solve(gram + regularizer, rhs)
                info = 0
                used_lambda = lambda_diag
                if solved is None:
                    used_lambda = lambda_diag * 10.0
                    solved = self._cholesky_solve(
                        gram + torch.diag(used_lambda).to(torch.complex64), rhs
                    )
                    info = 1
                if solved is None:
                    sqrt_weight = weight.sqrt().to(torch.complex64)
                    weighted_design = sqrt_weight.unsqueeze(-1) * design
                    weighted_response = sqrt_weight * response
                    augmented_design = torch.cat(
                        [
                            weighted_design,
                            torch.diag(used_lambda.clamp_min(0.0).sqrt()).to(torch.complex64),
                        ],
                        dim=0,
                    )
                    augmented_response = torch.cat(
                        [weighted_response, torch.zeros(28, device=phi.device, dtype=torch.complex64)]
                    )
                    solved = torch.linalg.lstsq(augmented_design, augmented_response).solution
                    info = 2

                fitted = design @ solved
                residual = response - fitted
                eig = torch.linalg.eigvalsh(gram).real.clamp_min(0.0)
                spectrum_sum = eig.sum().clamp_min(self.eps)
                probability = (eig / spectrum_sum).clamp_min(self.eps)
                effective_rank = torch.exp(-(probability * probability.log()).sum())
                condition = (eig[-1] + self.eps) / (eig[0] + self.eps)
                n_eff = weight.sum().square() / weight.square().sum().clamp_min(self.eps)
                nmse = (weight * residual.abs().square()).sum() / (
                    (weight * response.abs().square()).sum().clamp_min(self.eps)
                )
                system_diag = (gram.diagonal().real + used_lambda).clamp_min(self.eps)

                coefficients.append(solved)
                covariance.append(system_diag.reciprocal())
                infos.append(info)
                eigenvalues.append(eig)
                log_conditions.append(condition.log())
                effective_ranks.append(effective_rank)
                effective_samples.append(n_eff)
                coverages.append(coverage)
                nmses.append(nmse.real)

        return {
            "resp_coef": torch.stack(coefficients, dim=0),
            "resp_cov_diag": torch.stack(covariance, dim=0),
            "weights": torch.stack(normalized_weights, dim=0),
            "ridge_info": torch.tensor(infos, device=phi.device, dtype=torch.long),
            "resp_quality": {
                "gram_eigenvalues": torch.stack(eigenvalues, dim=0),
                "log_condition": torch.stack(log_conditions, dim=0),
                "effective_rank": torch.stack(effective_ranks, dim=0),
                "effective_sample_size": torch.stack(effective_samples, dim=0),
                "coverage": torch.stack(coverages, dim=0),
                "nmse": torch.stack(nmses, dim=0),
            },
        }


class SurfaceAnchorEncoder(nn.Module):
    """Evaluate a fitted response on one fixed 32-point complex excitation grid."""

    def __init__(self, response_basis: ResponseBasis, variance_temperature: float = 1.0):
        super().__init__()
        amplitudes = torch.tensor(
            [0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.05, 1.20], dtype=torch.float32
        )
        phases = torch.tensor([0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi])
        anchor_grid = (
            amplitudes[:, None] * torch.exp(torch.complex(torch.zeros_like(phases), phases))[None, :]
        ).reshape(1, 32).to(torch.complex64)
        with torch.no_grad():
            anchor_design = response_basis(anchor_grid).squeeze(0)
        self.register_buffer("anchor_grid", anchor_grid.squeeze(0), persistent=True)
        self.register_buffer("anchor_design", anchor_design, persistent=True)
        self.variance_temperature = float(variance_temperature)

    def forward(
        self,
        resp_coef: torch.Tensor,
        resp_cov_diag: torch.Tensor,
        coverage: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        design = self.anchor_design.to(device=resp_coef.device, dtype=torch.complex64)
        anchor = torch.einsum("qk,bk->bq", design, resp_coef.to(torch.complex64))
        variance = torch.einsum(
            "qk,bk->bq", design.abs().square().float(), resp_cov_diag.float()
        ).clamp_min(0.0)
        reliability = coverage.float().reshape(-1, 1).clamp(0.0, 1.0) * torch.exp(
            -variance / max(self.variance_temperature, 1e-6)
        )
        weighted_anchor = anchor * reliability.sqrt().to(torch.complex64)
        z_resp = torch.cat([weighted_anchor.real, weighted_anchor.imag], dim=1)
        z_resp = torch.nn.functional.normalize(z_resp.float(), dim=1, eps=1e-6)
        return anchor, z_resp, variance, reliability


class ResponseFusionGate(nn.Module):
    """Quality-only bounded residual gate; all quality inputs are detached."""

    def __init__(self, rho_max: float = 0.25, hidden: int = 8):
        super().__init__()
        rho_max = float(rho_max)
        if rho_max != 0.25:
            raise ValueError("ADV3B02-ECRS-V1 fixes rho_max=0.25")
        self.rho_max = rho_max
        self.active_rho_max = 0.0
        self.net = nn.Sequential(
            nn.Linear(7, int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def set_active_rho_max(self, value: float) -> None:
        self.active_rho_max = min(self.rho_max, max(0.0, float(value)))

    def forward(
        self,
        quality: Mapping[str, torch.Tensor],
        resp_cov_diag: torch.Tensor,
        sample_count: int,
    ) -> torch.Tensor:
        values = torch.stack(
            [
                quality["log_condition"],
                quality["effective_rank"] / 28.0,
                quality["effective_sample_size"] / float(max(1, sample_count)),
                quality["coverage"],
                torch.log1p(quality["nmse"].clamp_min(0.0)),
                quality["snr_db"] / 40.0,
                torch.log1p(resp_cov_diag.mean(dim=1).clamp_min(0.0)),
            ],
            dim=1,
        ).detach()
        return self.active_rho_max * torch.sigmoid(self.net(values).squeeze(-1))


class ResponseSurfaceBranch(nn.Module):
    """Parallel ADV3B02-ECRS-V1 local system-identification branch."""

    def __init__(
        self,
        identity_dim: int,
        *,
        response_basis_dim: int = 28,
        response_dim: int = 64,
        rho_max: float = 0.25,
        ridge_alpha: float = 1e-4,
    ):
        super().__init__()
        if int(identity_dim) != 160:
            raise ValueError("ADV3B02-ECRS-V1 fixes the existing identity dimension at 160")
        if int(response_basis_dim) != 28:
            raise ValueError("ADV3B02-ECRS-V1 fixes response_basis_dim=28")
        if int(response_dim) != 64:
            raise ValueError("ADV3B02-ECRS-V1 fixes response_dim=64")
        self.identity_dim = int(identity_dim)
        self.response_basis = ResponseBasis()
        self.nuisance_estimator = NuisanceEstimator()
        self.canonicalizer = AnalyticCanonicalizer()
        self.content_estimator = ContentEstimator()
        self.weighted_ridge = WeightedRidgeLayer(alpha_lambda=float(ridge_alpha))
        self.anchor_encoder = SurfaceAnchorEncoder(self.response_basis)
        self.fusion_gate = ResponseFusionGate(rho_max=float(rho_max))
        self.response_projection = nn.Linear(64, self.identity_dim, bias=False)
        nn.init.normal_(self.response_projection.weight, mean=0.0, std=1e-3)
        self.detach_identification_for_identity = True

    def forward(self, x: torch.Tensor, z_id_raw: torch.Tensor) -> Dict[str, object]:
        nuisance_coef = self.nuisance_estimator(x)
        canonical_iq = self.canonicalizer(x, nuisance_coef)
        s_hat, content_confidence = self.content_estimator(canonical_iq)
        design = self.response_basis(s_hat)
        ridge = self.weighted_ridge(
            design, iq_to_complex(canonical_iq), content_confidence
        )
        quality = dict(ridge["resp_quality"])
        quality["snr_db"] = -10.0 * torch.log10(quality["nmse"].clamp_min(1e-8))
        anchor, z_resp, anchor_variance, anchor_reliability = self.anchor_encoder(
            ridge["resp_coef"], ridge["resp_cov_diag"], quality["coverage"]
        )
        quality["anchor_variance"] = anchor_variance
        quality["anchor_reliability"] = anchor_reliability
        rho = self.fusion_gate(quality, ridge["resp_cov_diag"], int(x.size(-1)))
        identity_response = (
            z_resp.detach() if bool(self.detach_identification_for_identity) else z_resp
        )
        residual = self.response_projection(identity_response)
        z_resp_projected = torch.nn.functional.normalize(residual.float(), dim=1, eps=1e-6)
        z_id_fused = torch.nn.functional.normalize(
            z_id_raw.float() + rho.unsqueeze(1) * residual.float(), dim=1, eps=1e-6
        )
        return {
            "z_resp": z_resp,
            "z_id_fused": z_id_fused,
            "resp_coef": ridge["resp_coef"],
            "resp_cov_diag": ridge["resp_cov_diag"],
            "resp_quality": quality,
            "resp_anchor": anchor,
            "nuisance_coef": nuisance_coef,
            "content_confidence": content_confidence,
            "canonical_iq": canonical_iq,
            "s_hat": s_hat,
            "response_design": design,
            "response_weights": ridge["weights"],
            "ridge_info": ridge["ridge_info"],
            "rho_resp": rho,
            "z_resp_projected": z_resp_projected,
        }


class SatAnchorIdentityAdapter(nn.Module):
    """Zero-initialized low-rank identity residual and logit correction."""

    def __init__(self, feature_dim: int, num_classes: int, rank: int = 8):
        super().__init__()
        feature_dim = int(feature_dim)
        num_classes = int(num_classes)
        rank = int(rank)
        if min(feature_dim, num_classes, rank) < 1:
            raise ValueError("feature_dim, num_classes and rank must be positive")
        self.down = nn.Linear(feature_dim, rank, bias=False)
        self.up = nn.Linear(rank, feature_dim, bias=False)
        self.logit_correction = nn.Linear(feature_dim, num_classes, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.logit_correction.weight)

    def forward(
        self, feature: torch.Tensor, *, detach_backbone: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        source = feature.detach() if bool(detach_backbone) else feature
        residual = self.up(torch.nn.functional.gelu(self.down(source)))
        adapted = source + residual
        return adapted, self.logit_correction(adapted)


class RCNStatEncoder(nn.Module):
    """Lightweight receiver/channel/noise statistic encoder for the domain path."""

    def __init__(
        self,
        out_dim: int,
        hidden: Optional[int] = None,
        drop: float = 0.05,
        eps: float = 1e-6,
        mode: str = "full",
        stat_mode: Optional[str] = None,
    ):
        super().__init__()
        raw_mode = str(stat_mode if stat_mode is not None else mode or "full").lower().strip()
        aliases = {
            "18": "full",
            "full18": "full",
            "minimal6": "minimal_6",
            "minimal_6stats": "minimal_6",
            "minimal6stats": "minimal_6",
            "6stats": "minimal_6",
            "min6": "minimal_6",
        }
        self.mode = aliases.get(raw_mode, raw_mode)
        if self.mode not in ("full", "minimal_6"):
            raise ValueError("RCNStatEncoder mode must be one of: full,minimal_6")
        self.stat_dim = 6 if self.mode == "minimal_6" else 18
        hidden = int(hidden or max(64, out_dim // 2))
        self.eps = float(eps)
        self.net = nn.Sequential(
            nn.Linear(self.stat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(float(drop)),
            nn.Linear(hidden, out_dim),
        )

    def _moments(self, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        v = torch.nan_to_num(v.float(), nan=0.0, posinf=0.0, neginf=0.0)
        mean = v.mean(dim=1)
        std = v.std(dim=1, unbiased=False).clamp_min(self.eps)
        abs_mean = v.abs().mean(dim=1)
        return mean, std, abs_mean

    def _iq_stats(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
        i = x[:, 0, :]
        q = x[:, 1, :]
        amp = torch.sqrt(i * i + q * q + self.eps)
        power = torch.log1p(amp * amp)

        i_mean, i_std, i_abs = self._moments(i)
        q_mean, q_std, q_abs = self._moments(q)
        a_mean, a_std, a_abs = self._moments(amp)
        p_mean, p_std, _ = self._moments(power)
        iq_corr = ((i - i_mean[:, None]) * (q - q_mean[:, None])).mean(dim=1) / (i_std * q_std).clamp_min(self.eps)
        iq_imbalance = torch.log((i_std + self.eps) / (q_std + self.eps))

        if x.size(-1) > 1:
            di = i[:, 1:] - i[:, :-1]
            dq = q[:, 1:] - q[:, :-1]
            da = amp[:, 1:] - amp[:, :-1]
            # Phase increment avoids unstable unwrap while preserving CFO/phase-noise cues.
            cross = q[:, 1:] * i[:, :-1] - i[:, 1:] * q[:, :-1]
            dot = i[:, 1:] * i[:, :-1] + q[:, 1:] * q[:, :-1]
            dphi = torch.atan2(cross, dot + self.eps)
            di_abs = di.abs().mean(dim=1)
            dq_abs = dq.abs().mean(dim=1)
            da_abs = da.abs().mean(dim=1)
            dphi_mean, dphi_std, dphi_abs = self._moments(dphi)
        else:
            z = i.new_zeros(i.size(0))
            di_abs = dq_abs = da_abs = dphi_mean = dphi_std = dphi_abs = z

        dphi_summary = dphi_std + 0.1 * dphi_abs
        if self.mode == "minimal_6":
            return torch.stack(
                [
                    a_mean, a_std,
                    p_mean, p_std,
                    iq_corr.clamp(-5.0, 5.0), dphi_std,
                ],
                dim=1,
            )
        return torch.stack(
            [
                i_mean, i_std, i_abs,
                q_mean, q_std, q_abs,
                a_mean, a_std, a_abs,
                p_mean, p_std,
                iq_corr.clamp(-5.0, 5.0), iq_imbalance.clamp(-5.0, 5.0),
                di_abs, dq_abs, da_abs,
                dphi_mean.clamp(-math.pi, math.pi), dphi_summary,
            ],
            dim=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.size(1) != 2:
            raise ValueError("RCNStatEncoder expects IQ tensors shaped [B, 2, T]")
        return self.net(self._iq_stats(x))


class DomainFeatureEnhancer(nn.Module):
    """Fuse raw RCN statistics into the second/domain backbone feature."""

    def __init__(self, emb_dim: int, mode: str = "rcn_stats", strength: float = 0.35, drop: float = 0.05):
        super().__init__()
        self.mode = str(mode or "off").lower().strip()
        self.strength = float(strength)
        if self.mode not in ("off", "rcn_stats", "rcn_minimal_6stats"):
            raise ValueError("domain_enhancer must be one of: off,rcn_stats,rcn_minimal_6stats")
        if self.mode == "off" or self.strength <= 0.0:
            self.stat_encoder = None
            self.gate = None
            self.norm = nn.Identity()
        else:
            stat_mode = "minimal_6" if self.mode == "rcn_minimal_6stats" else "full"
            self.stat_encoder = RCNStatEncoder(emb_dim, hidden=max(64, emb_dim // 2), drop=drop, mode=stat_mode)
            self.gate = nn.Sequential(nn.Linear(2 * emb_dim, emb_dim), nn.Sigmoid())
            self.norm = nn.LayerNorm(emb_dim)

    def forward(self, z_dom: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.stat_encoder is None or self.gate is None:
            return z_dom, None
        z_rcn = self.stat_encoder(x)
        gate = self.gate(torch.cat([z_dom, z_rcn], dim=1))
        return self.norm(z_dom + self.strength * gate * z_rcn), z_rcn


class ParameterMatchedIdentityCapacity(nn.Module):
    """Move the removed domain-branch budget into the identity embedding."""

    def __init__(self, emb_dim: int, num_classes: int, target_params: int):
        super().__init__()
        self.emb_dim = int(emb_dim)
        self.num_classes = int(num_classes)
        self.target_params = int(target_params)
        head_params = self.emb_dim * self.num_classes + self.num_classes
        if self.target_params <= head_params:
            raise ValueError("parameter-matched identity budget is too small")
        remaining = self.target_params - head_params
        per_hidden = 2 * self.emb_dim + 1
        hidden = max(1, (remaining - self.emb_dim) // per_hidden)
        while hidden > 0 and per_hidden * hidden + self.emb_dim > remaining:
            hidden -= 1
        if hidden <= 0:
            raise ValueError("parameter-matched identity residual has no capacity")
        self.hidden = int(hidden)
        self.fc1 = nn.Linear(self.emb_dim, self.hidden)
        self.fc2 = nn.Linear(self.hidden, self.emb_dim)
        self.tx_correction = nn.Linear(self.emb_dim, self.num_classes)
        used = sum(parameter.numel() for parameter in self.parameters())
        tail_count = self.target_params - int(used)
        if tail_count < 0:
            raise ValueError("parameter-matched identity budget underflow")
        self.tail = nn.Parameter(torch.zeros(tail_count))
        actual = sum(parameter.numel() for parameter in self.parameters())
        if actual != self.target_params:
            raise RuntimeError(
                f"parameter-matched identity budget drift: {actual} != {self.target_params}"
            )

    def forward(self, z_id: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = self.fc2(torch.nn.functional.gelu(self.fc1(z_id)))
        adapted = z_id + residual
        if self.tail.numel() > 0:
            indices = torch.arange(
                self.emb_dim,
                device=adapted.device,
                dtype=torch.long,
            ) % int(self.tail.numel())
            gate = self.tail[indices]
            if self.tail.numel() > self.emb_dim:
                gate = gate + self.tail[self.emb_dim :].mean()
            adapted = adapted + 0.01 * torch.tanh(gate).unsqueeze(0) * adapted
        return adapted, self.tx_correction(adapted)


class FeatureBackboneAdapter(nn.Module):
    """Expose common RFFI backbones through the CVS-RFFI auxiliary-output API."""

    def __init__(
        self,
        family: str,
        num_classes: int,
        input_len: int = 256,
        sample_rate_hz: float = 25e6,
        drop: float = 0.0,
    ):
        super().__init__()
        self.arch_family = str(family or "cvsincnet").lower().strip()
        self.input_len = int(input_len)
        if self.arch_family == "resnet18_1d":
            if ResNet1DEncoder is None or MLPClassifier is None:
                raise ImportError("resnet18_1d requires baselines.common.resnet1d")
            self.encoder = ResNet1DEncoder(
                input_channels=2,
                embedding_dim=512,
                dropout=float(drop),
                use_projection=True,
            )
            self.emb_dim = 512
            self.classifier = MLPClassifier(self.emb_dim, int(num_classes), hidden_dim=256, dropout=float(drop))
        elif self.arch_family in {"cvcnn", "sinc_cvcnn"}:
            model_cls = SincCVCNN if self.arch_family == "sinc_cvcnn" else BasicCVCNN
            if model_cls is None:
                raise ImportError(f"{self.arch_family} requires baselines.cvcnn_ce.model")
            self.encoder = model_cls(
                num_classes=int(num_classes),
                input_len=int(input_len),
                base_channels=32,
                embedding_dim=128,
                dropout=float(drop),
                sample_rate_hz=float(sample_rate_hz),
            ) if self.arch_family == "sinc_cvcnn" else model_cls(
                num_classes=int(num_classes),
                input_len=int(input_len),
                base_channels=32,
                embedding_dim=128,
                dropout=float(drop),
            )
            self.emb_dim = 128
            self.classifier = None
        else:
            raise ValueError(
                "arch_family must be one of: cvsincnet,resnet18_1d,cvcnn,sinc_cvcnn; "
                f"got {family!r}"
            )

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        return_aux: bool = False,
        domain_labels: Optional[torch.Tensor] = None,
    ):
        del y, domain_labels
        if self.arch_family == "resnet18_1d":
            z = self.encoder(x)
            logits = self.classifier(z)
        else:
            z = self.encoder.forward_features(x)
            logits = self.encoder.classifier(z)
        if not return_aux:
            return logits

        zero_feat = torch.zeros_like(z)
        zero_score = z.new_zeros(z.size(0))
        return {
            "logits": logits,
            "feat_cls": z,
            "feat_imp": z,
            "feat_dac": zero_feat,
            "feat_pa": zero_feat,
            "feat_joint": z,
            "feat_con": z,
            "base": z,
            "dac_pred": zero_score,
            "pa_pred": zero_score,
            "arch_family": self.arch_family,
        }


def build_arch_backbone(
    arch_family: str,
    *,
    num_classes: int,
    model_size: str,
    dataset: str,
    input_len: int,
    sample_rate_hz: float,
    model_variant: str,
    branch_ablation: str,
    mixstyle_on: bool = False,
    mixstyle_p: float = 0.3,
    mixstyle_alpha: float = 0.1,
    mixstyle_eps: float = 1e-6,
    mixstyle_layers: str = "time_down,t1",
    mixstyle_use_domain_label: bool = True,
    mixstyle_mix: str = "crossdomain",
    mixstyle_strength: float = 1.0,
    mixstyle_fallback: str = "random",
    use_circularity: bool = True,
    use_freq_stats: bool = True,
    use_pa_stats: bool = True,
    use_freq_band_gate: bool = True,
    freq_feature_source: str = "raw_fft",
    pa_feature_source: str = "raw_iq",
    pa_orders: Optional[Sequence[int]] = None,
    use_aux_spectral_stats: bool = True,
    channel_trim_scale: float = 1.0,
    time_stability_mode: str = "off",
    freq_stability_mode: str = "off",
    time_stability_channels: int = 8,
    freq_stability_channels: int = 4,
    use_crra: bool = False,
    crra_rank: int = 8,
    crra_alpha_max: float = 0.25,
    crra_shrinkage: float = 0.10,
    crra_condition_dim: int = 32,
    crra_nuisance_dim: int = 9,
    crra_start_epoch: int = 17,
    crra_ramp_epochs: int = 30,
) -> nn.Module:
    family = str(arch_family or "cvsincnet").lower().strip()
    if family == "cvsincnet":
        return build_single_model_compat(
            num_classes=num_classes,
            model_size=model_size,
            dataset=dataset,
            input_len=input_len,
            sample_rate_hz=sample_rate_hz,
            model_variant=model_variant,
            branch_ablation=branch_ablation,
            mixstyle_on=mixstyle_on,
            mixstyle_p=float(mixstyle_p),
            mixstyle_alpha=float(mixstyle_alpha),
            mixstyle_eps=float(mixstyle_eps),
            mixstyle_layers=str(mixstyle_layers),
            mixstyle_use_domain_label=bool(mixstyle_use_domain_label),
            mixstyle_mix=str(mixstyle_mix),
            mixstyle_strength=float(mixstyle_strength),
            mixstyle_fallback=str(mixstyle_fallback),
            use_circularity=bool(use_circularity),
            use_freq_stats=bool(use_freq_stats),
            use_pa_stats=bool(use_pa_stats),
            use_freq_band_gate=bool(use_freq_band_gate),
            freq_feature_source=str(freq_feature_source),
            pa_feature_source=str(pa_feature_source),
            pa_orders=pa_orders,
            use_aux_spectral_stats=bool(use_aux_spectral_stats),
            channel_trim_scale=float(channel_trim_scale),
            time_stability_mode=time_stability_mode,
            freq_stability_mode=freq_stability_mode,
            time_stability_channels=int(time_stability_channels),
            freq_stability_channels=int(freq_stability_channels),
            use_crra=bool(use_crra),
            crra_rank=int(crra_rank),
            crra_alpha_max=float(crra_alpha_max),
            crra_shrinkage=float(crra_shrinkage),
            crra_condition_dim=int(crra_condition_dim),
            crra_nuisance_dim=int(crra_nuisance_dim),
            crra_start_epoch=int(crra_start_epoch),
            crra_ramp_epochs=int(crra_ramp_epochs),
        )
    return FeatureBackboneAdapter(
        family,
        num_classes=int(num_classes),
        input_len=int(input_len),
        sample_rate_hz=float(sample_rate_hz),
        drop=0.0,
    )


class DualCVSincNetDisentangle(nn.Module):
    """
    双 CV-SincNet 解耦模型。

    这版与原版相比不改 state_dict 结构，只额外在 return_aux=True 时
    暴露 id/dom 分支的关键中间特征，便于训练脚本做：
      - PA robustness consistency on id_feat_joint / id_feat_imp
      - DAC / PA strength supervision
      - branch-selective sensitivity ranking
    """
    def __init__(
        self,
        num_classes: int,
        num_domains: int,
        model_size: str = "S",
        dataset: str = "wisig",
        input_len: int = 256,
        sample_rate_hz: float = 25e6,
        drop: float = 0.1,
        id_feature_key: str = "feat_joint",
        dom_feature_key: str = "feat_imp",
        model_variant: str = "base",
        branch_ablation: str = "none",
        mixstyle_on: bool = False,
        mixstyle_p: float = 0.3,
        mixstyle_alpha: float = 0.1,
        mixstyle_eps: float = 1e-6,
        mixstyle_layers: str = "time_down,t1",
        mixstyle_use_domain_label: bool = True,
        mixstyle_mix: str = "crossdomain",
        mixstyle_strength: float = 1.0,
        mixstyle_fallback: str = "random",
        domain_branch_ablation: str = "same",
        domain_enhancer: str = "rcn_stats",
        domain_enhancer_strength: float = 0.35,
        use_circularity: bool = True,
        use_freq_stats: bool = True,
        use_pa_stats: bool = True,
        use_freq_band_gate: bool = True,
        freq_feature_source: str = "raw_fft",
        pa_feature_source: str = "raw_iq",
        pa_orders: Optional[Sequence[int]] = None,
        use_aux_spectral_stats: bool = True,
        channel_trim_scale: float = 1.0,
        id_time_stability_mode: str = "off",
        id_freq_stability_mode: str = "off",
        domain_time_stability_mode: str = "off",
        domain_freq_stability_mode: str = "off",
        time_stability_channels: int = 8,
        freq_stability_channels: int = 4,
        fast_infer_when_no_aux: bool = True,
        use_tx_adv_on_zdom: bool = False,
        arch_family: str = "cvsincnet",
        representation_mode: str = "dual",
        use_crra: bool = False,
        crra_rank: int = 8,
        crra_alpha_max: float = 0.25,
        crra_shrinkage: float = 0.10,
        crra_condition_dim: int = 32,
        crra_nuisance_dim: int = 9,
        crra_start_epoch: int = 17,
        crra_ramp_epochs: int = 30,
        sat_anchor_adapter: bool = False,
        sat_anchor_adapter_rank: int = 8,
        use_ecrs: bool = False,
        ecrs_config: Optional[Mapping[str, object]] = None,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_domains = int(max(1, num_domains))
        self.arch_family = str(arch_family or "cvsincnet").lower().strip()
        self.representation_mode = str(representation_mode or "dual").lower().strip()
        if self.representation_mode not in {"dual", "single_parameter_matched"}:
            raise ValueError(
                "representation_mode must be dual or single_parameter_matched"
            )
        self.id_feature_key = str(id_feature_key)
        self.dom_feature_key = str(dom_feature_key)
        self.model_variant = str(model_variant or "base").lower().strip()
        self.branch_ablation = str(branch_ablation or "none")
        self.domain_branch_ablation = (
            self.branch_ablation
            if str(domain_branch_ablation or "same").lower().strip() in ("", "same")
            else str(domain_branch_ablation)
        )
        self.mixstyle_on = bool(mixstyle_on)
        self.fast_infer_when_no_aux = bool(fast_infer_when_no_aux)
        self.use_tx_adv_on_zdom = bool(use_tx_adv_on_zdom)
        self.use_crra = bool(use_crra)
        self.use_ecrs = bool(use_ecrs)
        self.ecrs_config = dict(ecrs_config or {})
        # The response branch is populated only on the opt-in route. Keeping a
        # plain None here is deliberate: the legacy route gains no parameters
        # and therefore preserves strict state_dict compatibility.
        self.ecrs = None
        self.crra_epoch = 1
        self.id_time_stability_mode = str(id_time_stability_mode or "off").lower().strip()
        self.id_freq_stability_mode = str(id_freq_stability_mode or "off").lower().strip()
        self.domain_time_stability_mode = self._resolve_domain_stability_mode(
            domain_time_stability_mode,
            self.id_time_stability_mode,
        )
        self.domain_freq_stability_mode = self._resolve_domain_stability_mode(
            domain_freq_stability_mode,
            self.id_freq_stability_mode,
        )

        self.id_backbone = build_arch_backbone(
            self.arch_family,
            num_classes=num_classes,
            model_size=model_size,
            dataset=dataset,
            input_len=input_len,
            sample_rate_hz=sample_rate_hz,
            model_variant=self.model_variant,
            branch_ablation=self.branch_ablation,
            mixstyle_on=self.mixstyle_on,
            mixstyle_p=float(mixstyle_p),
            mixstyle_alpha=float(mixstyle_alpha),
            mixstyle_eps=float(mixstyle_eps),
            mixstyle_layers=str(mixstyle_layers),
            mixstyle_use_domain_label=bool(mixstyle_use_domain_label),
            mixstyle_mix=str(mixstyle_mix),
            mixstyle_strength=float(mixstyle_strength),
            mixstyle_fallback=str(mixstyle_fallback),
            use_circularity=bool(use_circularity),
            use_freq_stats=bool(use_freq_stats),
            use_pa_stats=bool(use_pa_stats),
            use_freq_band_gate=bool(use_freq_band_gate),
            freq_feature_source=str(freq_feature_source),
            pa_feature_source=str(pa_feature_source),
            pa_orders=pa_orders,
            use_aux_spectral_stats=bool(use_aux_spectral_stats),
            channel_trim_scale=float(channel_trim_scale),
            time_stability_mode=self.id_time_stability_mode,
            freq_stability_mode=self.id_freq_stability_mode,
            time_stability_channels=int(time_stability_channels),
            freq_stability_channels=int(freq_stability_channels),
            use_crra=bool(use_crra),
            crra_rank=int(crra_rank),
            crra_alpha_max=float(crra_alpha_max),
            crra_shrinkage=float(crra_shrinkage),
            crra_condition_dim=int(crra_condition_dim),
            crra_nuisance_dim=int(crra_nuisance_dim),
            crra_start_epoch=int(crra_start_epoch),
            crra_ramp_epochs=int(crra_ramp_epochs),
        )
        self.dom_backbone = build_arch_backbone(
            self.arch_family,
            num_classes=num_classes,
            model_size=model_size,
            dataset=dataset,
            input_len=input_len,
            sample_rate_hz=sample_rate_hz,
            model_variant=self.model_variant,
            branch_ablation=self.domain_branch_ablation,
            mixstyle_on=False,
            use_circularity=bool(use_circularity),
            use_freq_stats=bool(use_freq_stats),
            use_pa_stats=bool(use_pa_stats),
            use_freq_band_gate=bool(use_freq_band_gate),
            freq_feature_source=str(freq_feature_source),
            pa_feature_source=str(pa_feature_source),
            pa_orders=pa_orders,
            use_aux_spectral_stats=bool(use_aux_spectral_stats),
            channel_trim_scale=float(channel_trim_scale),
            time_stability_mode=self.domain_time_stability_mode,
            freq_stability_mode=self.domain_freq_stability_mode,
            time_stability_channels=int(time_stability_channels),
            freq_stability_channels=int(freq_stability_channels),
            use_crra=False,
        )
        if self.arch_family == "cvsincnet" and self.model_variant in {"lite_b", "lite_d", "lite_e", "lite_f", "lite_g", "lite_h"}:
            self._share_early_stem()

        self.emb_dim = self._infer_emb_dim(self.id_backbone)
        if self.use_ecrs:
            if self.representation_mode != "dual":
                raise ValueError("ADV3B02-ECRS-V1 requires the existing dual ADV3B02 backbone")
            allowed_ecrs_keys = {
                "response_basis_dim",
                "response_dim",
                "rho_max",
                "ridge_alpha",
            }
            unknown_ecrs_keys = sorted(set(self.ecrs_config) - allowed_ecrs_keys)
            if unknown_ecrs_keys:
                raise ValueError(f"unsupported ECRS-V1 config keys: {unknown_ecrs_keys}")
            self.ecrs = ResponseSurfaceBranch(
                self.emb_dim,
                response_basis_dim=int(self.ecrs_config.get("response_basis_dim", 28)),
                response_dim=int(self.ecrs_config.get("response_dim", 64)),
                rho_max=float(self.ecrs_config.get("rho_max", 0.25)),
                ridge_alpha=float(self.ecrs_config.get("ridge_alpha", 1e-4)),
            )
        self.sat_anchor_identity_adapter = (
            SatAnchorIdentityAdapter(
                self.emb_dim,
                self.num_classes,
                rank=int(sat_anchor_adapter_rank),
            )
            if bool(sat_anchor_adapter)
            else None
        )
        self.dom_enhancer = DomainFeatureEnhancer(
            self.emb_dim,
            mode=str(domain_enhancer),
            strength=float(domain_enhancer_strength),
            drop=drop * 0.5,
        )
        self.dom_head = MLPHead(self.emb_dim, self.num_domains, hidden=max(64, self.emb_dim // 2), drop=drop)
        self.adv_head = MLPHead(self.emb_dim, self.num_domains, hidden=max(64, self.emb_dim // 2), drop=drop)
        self.tx_adv_head = (
            MLPHead(self.emb_dim, self.num_classes, hidden=max(64, self.emb_dim // 2), drop=drop)
            if self.use_tx_adv_on_zdom
            else None
        )
        self.crra_condition_tx_adv_head = (
            MLPHead(
                int(crra_condition_dim),
                self.num_classes,
                hidden=max(64, int(crra_condition_dim) // 2),
                drop=drop,
            )
            if self.use_crra
            else None
        )
        self.identity_capacity = None
        if self.representation_mode == "single_parameter_matched":
            replaced_modules = (
                self.dom_backbone,
                self.dom_enhancer,
                self.dom_head,
                self.adv_head,
                self.tx_adv_head,
                self.crra_condition_tx_adv_head,
            )
            retained_parameter_ids = {
                id(parameter) for parameter in self.id_backbone.parameters()
            }
            counted_parameter_ids = set()
            target_params = 0
            for module in replaced_modules:
                if module is None:
                    continue
                for parameter in module.parameters():
                    parameter_id = id(parameter)
                    if (
                        parameter_id in retained_parameter_ids
                        or parameter_id in counted_parameter_ids
                    ):
                        continue
                    counted_parameter_ids.add(parameter_id)
                    target_params += int(parameter.numel())
            self.identity_capacity = ParameterMatchedIdentityCapacity(
                self.emb_dim,
                self.num_classes,
                target_params,
            )
            self.dom_backbone = None
            self.dom_enhancer = None
            self.dom_head = None
            self.adv_head = None
            self.tx_adv_head = None
            self.crra_condition_tx_adv_head = None

    def set_crra_epoch(self, epoch: int) -> None:
        self.crra_epoch = max(1, int(epoch))
        for backbone in (self.id_backbone, self.dom_backbone):
            if backbone is not None and hasattr(backbone, "set_crra_epoch"):
                backbone.set_crra_epoch(self.crra_epoch)

    def _share_early_stem(self) -> None:
        """Share the lowest-level IQ/filterbank stem only for Lite-B.

        This keeps output semantics unchanged and avoids tying the later ID/domain
        representation blocks, which would make ablations harder to interpret.
        """
        for name in ("sinc", "hf"):
            if hasattr(self.id_backbone, name) and hasattr(self.dom_backbone, name):
                src = getattr(self.id_backbone, name)
                dst = getattr(self.dom_backbone, name)
                if src is not None and dst is not None:
                    setattr(self.dom_backbone, name, src)

    @staticmethod
    def _resolve_domain_stability_mode(domain_mode: str, id_mode: str) -> str:
        mode = str(domain_mode or "off").lower().strip()
        if mode in ("", "same", "match_id"):
            return str(id_mode or "off").lower().strip()
        return mode

    @staticmethod
    def _infer_emb_dim(backbone: nn.Module) -> int:
        cls_head = getattr(backbone, "cls_head", None)
        if cls_head is not None:
            head = getattr(cls_head, "head", None)
            weight = getattr(head, "weight", None)
            if weight is not None and hasattr(weight, "shape") and len(weight.shape) == 2:
                return int(weight.shape[1])
        for name in ("emb_dim", "embed_dim"):
            if hasattr(backbone, name):
                return int(getattr(backbone, name))
        return 256

    @staticmethod
    def _pick_from_keys(aux: Dict[str, torch.Tensor], preferred_key: Optional[str], fallback_keys) -> torch.Tensor:
        keys = []
        if preferred_key is not None and str(preferred_key).strip() != "":
            keys.append(str(preferred_key))
        keys.extend([k for k in fallback_keys if k not in keys])
        for key in keys:
            v = aux.get(key, None)
            if torch.is_tensor(v):
                return v
        raise KeyError(f"Cannot find feature from keys={keys}; available={list(aux.keys())}")

    def _pick_z_id(self, aux: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self._pick_from_keys(aux, self.id_feature_key, ("feat_joint", "feat_cls", "feat_con", "base"))

    def _pick_z_dom(self, aux: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self._pick_from_keys(aux, self.dom_feature_key, ("feat_imp", "feat_pa", "feat_dac", "base", "feat_con", "feat_cls", "feat_joint"))

    def _classify_identity_feature(
        self,
        feature: torch.Tensor,
        labels: Optional[torch.Tensor],
    ) -> torch.Tensor:
        classifier = getattr(getattr(self.id_backbone, "cls_head", None), "head", None)
        if classifier is None or not callable(classifier):
            raise RuntimeError("ECRS requires the existing ADV3B02 CosFace identity head")
        return classifier(feature, labels=labels)

    def forward(
        self,
        x: torch.Tensor,
        y_tx: Optional[torch.Tensor] = None,
        grl_lambda: float = 1.0,
        return_aux: bool = False,
        domain_labels: Optional[torch.Tensor] = None,
        crra_epoch: Optional[int] = None,
        update_crra_support: bool = False,
        crra_support_mask: Optional[torch.Tensor] = None,
        sat_anchor_detach_backbone: bool = False,
    ):
        if self.representation_mode == "single_parameter_matched":
            aux_id = backbone_forward_compat(
                self.id_backbone,
                x,
                y=y_tx,
                return_aux=True,
                domain_labels=domain_labels,
                crra_epoch=crra_epoch,
                update_crra_support=update_crra_support,
                crra_support_mask=crra_support_mask,
            )
            base_logits = aux_id["logits"]
            base_z_id = self._pick_z_id(aux_id)
            z_id, correction = self.identity_capacity(base_z_id)
            tx_logits = base_logits + correction
            if self.sat_anchor_identity_adapter is not None:
                z_id, sat_correction = self.sat_anchor_identity_adapter(
                    z_id,
                    detach_backbone=bool(sat_anchor_detach_backbone),
                )
                base_tx_logits = (
                    tx_logits.detach()
                    if bool(sat_anchor_detach_backbone)
                    else tx_logits
                )
                tx_logits = base_tx_logits + sat_correction
            if not return_aux:
                return tx_logits
            zero_domain_logits = tx_logits.new_zeros(
                (int(tx_logits.size(0)), self.num_domains)
            )
            return {
                "tx_logits": tx_logits,
                "dom_logits": zero_domain_logits,
                "adv_dom_logits": zero_domain_logits,
                "z_id": z_id,
                "z_dom": z_id,
                "z_dom_raw": z_id,
                "z_id_key": self.id_feature_key,
                "z_dom_key": "shared_identity_no_domain_representation",
                "representation_mode": self.representation_mode,
                "domain_branch_ablation": "not_present",
                "id_time_stability_mode": self.id_time_stability_mode,
                "id_freq_stability_mode": self.id_freq_stability_mode,
                "domain_time_stability_mode": "not_present",
                "domain_freq_stability_mode": "not_present",
                "tx_adv_on_zdom": False,
                "crra_condition_tx_adv_logits": None,
                "aux_id": aux_id,
                "aux_dom": {},
            }

        if (
            (not return_aux)
            and self.fast_infer_when_no_aux
            and self.sat_anchor_identity_adapter is None
            and not self.use_ecrs
        ):
            return backbone_forward_compat(
                self.id_backbone,
                x,
                y=y_tx,
                return_aux=False,
                domain_labels=domain_labels,
                crra_epoch=crra_epoch,
                update_crra_support=update_crra_support,
                crra_support_mask=crra_support_mask,
            )

        aux_id = backbone_forward_compat(
            self.id_backbone,
            x,
            y=y_tx,
            return_aux=True,
            domain_labels=domain_labels,
            crra_epoch=crra_epoch,
            update_crra_support=update_crra_support,
            crra_support_mask=crra_support_mask,
        )
        aux_dom = backbone_forward_compat(
            self.dom_backbone,
            x,
            y=None,
            return_aux=True,
            domain_labels=None,
            crra_epoch=crra_epoch,
            update_crra_support=False,
            crra_support_mask=None,
        )

        tx_logits = aux_id["logits"]
        tx_logits_raw = tx_logits
        z_id = self._pick_z_id(aux_id)
        if self.sat_anchor_identity_adapter is not None:
            z_id, sat_correction = self.sat_anchor_identity_adapter(
                z_id,
                detach_backbone=bool(sat_anchor_detach_backbone),
            )
            base_tx_logits = (
                tx_logits.detach()
                if bool(sat_anchor_detach_backbone)
                else tx_logits
            )
            tx_logits = base_tx_logits + sat_correction
            tx_logits_raw = tx_logits
        z_id_raw = z_id
        ecrs_out = None
        if self.ecrs is not None:
            ecrs_out = self.ecrs(x, z_id_raw)
            z_id = ecrs_out["z_id_fused"]
            tx_logits = self._classify_identity_feature(z_id, y_tx)
            ecrs_out["resp_tx_logits"] = self._classify_identity_feature(
                ecrs_out["z_resp_projected"], y_tx
            )
        z_dom_raw = self._pick_z_dom(aux_dom)
        z_dom, z_dom_rcn = self.dom_enhancer(z_dom_raw, x)

        dom_logits = self.dom_head(z_dom)
        adv_dom_logits = self.adv_head(grad_reverse(z_id, grl_lambda))
        tx_adv_logits = self.tx_adv_head(grad_reverse(z_dom, grl_lambda)) if self.tx_adv_head is not None else None
        crra_q_raw = aux_id.get("crra_q_raw", None)
        crra_condition_tx_adv_logits = (
            self.crra_condition_tx_adv_head(grad_reverse(crra_q_raw, grl_lambda))
            if self.crra_condition_tx_adv_head is not None and torch.is_tensor(crra_q_raw)
            else None
        )

        if not return_aux:
            return tx_logits

        out = {
            "tx_logits": tx_logits,
            "dom_logits": dom_logits,
            "adv_dom_logits": adv_dom_logits,
            "z_id": z_id,
            "z_dom": z_dom,
            "z_dom_raw": z_dom_raw,
            "z_id_key": self.id_feature_key,
            "z_dom_key": self.dom_feature_key,
            "domain_branch_ablation": self.domain_branch_ablation,
            "id_time_stability_mode": self.id_time_stability_mode,
            "id_freq_stability_mode": self.id_freq_stability_mode,
            "domain_time_stability_mode": self.domain_time_stability_mode,
            "domain_freq_stability_mode": self.domain_freq_stability_mode,
            "tx_adv_on_zdom": self.tx_adv_head is not None,
            "crra_condition_tx_adv_on_q": self.crra_condition_tx_adv_head is not None,
            "representation_mode": self.representation_mode,
            "aux_id": aux_id,
            "aux_dom": aux_dom,
            "crra_enabled": bool(self.use_crra),
            "crra_correction_energy": aux_id.get(
                "crra_correction_energy", x.new_zeros((x.size(0),))
            ),
            "crra_gate": aux_id.get("crra_gate", x.new_zeros((x.size(0),))),
            "crra_alpha": aux_id.get("crra_alpha", x.new_zeros((x.size(0),))),
            "crra_support_distance": aux_id.get(
                "crra_support_distance", x.new_zeros((x.size(0),))
            ),
            "crra_q": aux_id.get("crra_q", None),
            "crra_q_raw": aux_id.get("crra_q_raw", None),
            "crra_nuisance_pred": aux_id.get("crra_nuisance_pred", None),
            "crra_condition_tx_adv_logits": crra_condition_tx_adv_logits,
        }
        if ecrs_out is not None:
            out.update(ecrs_out)
            out["tx_logits_raw"] = tx_logits_raw
            out["z_id_raw"] = z_id_raw
            out["z_id_fused"] = z_id
            out["z_id"] = z_id
        if torch.is_tensor(tx_adv_logits):
            out["tx_adv_logits"] = tx_adv_logits
        if torch.is_tensor(crra_condition_tx_adv_logits):
            out["crra_condition_tx_adv_logits"] = crra_condition_tx_adv_logits
        if torch.is_tensor(z_dom_rcn):
            out["z_dom_rcn"] = z_dom_rcn

        # top-level aliases: no parameter changes, only easier access
        alias_map = {
            "id_feat_cls": ("aux_id", "feat_cls"),
            "id_feat_imp": ("aux_id", "feat_imp"),
            "id_feat_dac": ("aux_id", "feat_dac"),
            "id_feat_pa": ("aux_id", "feat_pa"),
            "id_feat_joint": ("aux_id", "feat_joint"),
            "id_feat_con": ("aux_id", "feat_con"),
            "id_base": ("aux_id", "base"),
            "id_dac_pred": ("aux_id", "dac_pred"),
            "id_pa_pred": ("aux_id", "pa_pred"),
            "dom_feat_cls": ("aux_dom", "feat_cls"),
            "dom_feat_imp": ("aux_dom", "feat_imp"),
            "dom_feat_dac": ("aux_dom", "feat_dac"),
            "dom_feat_pa": ("aux_dom", "feat_pa"),
            "dom_feat_joint": ("aux_dom", "feat_joint"),
            "dom_feat_con": ("aux_dom", "feat_con"),
            "dom_base": ("aux_dom", "base"),
            "dom_dac_pred": ("aux_dom", "dac_pred"),
            "dom_pa_pred": ("aux_dom", "pa_pred"),
        }
        for name, (g, k) in alias_map.items():
            v = out[g].get(k, None)
            if torch.is_tensor(v):
                out[name] = v
        return out


def build_dual_model(
    num_classes: int,
    num_domains: int,
    model_size: str = "S",
    dataset: str = "wisig",
    input_len: int = 256,
    sample_rate_hz: float = 25e6,
    id_feature_key: str = "feat_joint",
    dom_feature_key: str = "feat_imp",
    model_variant: str = "base",
    branch_ablation: str = "none",
    mixstyle_on: bool = False,
    mixstyle_p: float = 0.3,
    mixstyle_alpha: float = 0.1,
    mixstyle_eps: float = 1e-6,
    mixstyle_layers: str = "time_down,t1",
    mixstyle_use_domain_label: bool = True,
    mixstyle_mix: str = "crossdomain",
    mixstyle_strength: float = 1.0,
    mixstyle_fallback: str = "random",
    domain_branch_ablation: str = "same",
    domain_enhancer: str = "rcn_stats",
    domain_enhancer_strength: float = 0.35,
    use_circularity: bool = True,
    use_freq_stats: bool = True,
    use_pa_stats: bool = True,
    use_freq_band_gate: bool = True,
    freq_feature_source: str = "raw_fft",
    pa_feature_source: str = "raw_iq",
    pa_orders: Optional[Sequence[int]] = None,
    use_aux_spectral_stats: bool = True,
    channel_trim_scale: float = 1.0,
    id_time_stability_mode: str = "off",
    id_freq_stability_mode: str = "off",
    domain_time_stability_mode: str = "off",
    domain_freq_stability_mode: str = "off",
    time_stability_channels: int = 8,
    freq_stability_channels: int = 4,
    fast_infer_when_no_aux: bool = True,
    use_tx_adv_on_zdom: bool = False,
    arch_family: str = "cvsincnet",
    representation_mode: str = "dual",
    use_crra: bool = False,
    crra_rank: int = 8,
    crra_alpha_max: float = 0.25,
    crra_shrinkage: float = 0.10,
    crra_condition_dim: int = 32,
    crra_nuisance_dim: int = 9,
    crra_start_epoch: int = 17,
    crra_ramp_epochs: int = 30,
    sat_anchor_adapter: bool = False,
    sat_anchor_adapter_rank: int = 8,
    use_ecrs: bool = False,
    ecrs_config: Optional[Mapping[str, object]] = None,
) -> DualCVSincNetDisentangle:
    return DualCVSincNetDisentangle(
        num_classes=num_classes,
        num_domains=num_domains,
        model_size=model_size,
        dataset=dataset,
        input_len=input_len,
        sample_rate_hz=sample_rate_hz,
        id_feature_key=id_feature_key,
        dom_feature_key=dom_feature_key,
        model_variant=model_variant,
        branch_ablation=branch_ablation,
        mixstyle_on=mixstyle_on,
        mixstyle_p=mixstyle_p,
        mixstyle_alpha=mixstyle_alpha,
        mixstyle_eps=mixstyle_eps,
        mixstyle_layers=mixstyle_layers,
        mixstyle_use_domain_label=mixstyle_use_domain_label,
        mixstyle_mix=mixstyle_mix,
        mixstyle_strength=mixstyle_strength,
        mixstyle_fallback=mixstyle_fallback,
        domain_branch_ablation=domain_branch_ablation,
        domain_enhancer=domain_enhancer,
        domain_enhancer_strength=domain_enhancer_strength,
        use_circularity=use_circularity,
        use_freq_stats=use_freq_stats,
        use_pa_stats=use_pa_stats,
        use_freq_band_gate=use_freq_band_gate,
        freq_feature_source=freq_feature_source,
        pa_feature_source=pa_feature_source,
        pa_orders=pa_orders,
        use_aux_spectral_stats=use_aux_spectral_stats,
        channel_trim_scale=channel_trim_scale,
        id_time_stability_mode=id_time_stability_mode,
        id_freq_stability_mode=id_freq_stability_mode,
        domain_time_stability_mode=domain_time_stability_mode,
        domain_freq_stability_mode=domain_freq_stability_mode,
        time_stability_channels=time_stability_channels,
        freq_stability_channels=freq_stability_channels,
        fast_infer_when_no_aux=fast_infer_when_no_aux,
        use_tx_adv_on_zdom=use_tx_adv_on_zdom,
        arch_family=arch_family,
        representation_mode=representation_mode,
        use_crra=use_crra,
        crra_rank=crra_rank,
        crra_alpha_max=crra_alpha_max,
        crra_shrinkage=crra_shrinkage,
        crra_condition_dim=crra_condition_dim,
        crra_nuisance_dim=crra_nuisance_dim,
        crra_start_epoch=crra_start_epoch,
        crra_ramp_epochs=crra_ramp_epochs,
        sat_anchor_adapter=sat_anchor_adapter,
        sat_anchor_adapter_rank=sat_anchor_adapter_rank,
        use_ecrs=use_ecrs,
        ecrs_config=ecrs_config,
    )
