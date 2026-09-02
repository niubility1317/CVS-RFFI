
from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from cvsrffi.phase1_fcr_canonicalizer import ConservativeCanonicalizer
from cvsrffi.phase1_fcr_decoder import PhysicsOrderedDecoder
from cvsrffi.phase1_fcr_factors import ContentFactorEncoder, excitation_features
from cvsrffi.phase1_fcr_fingerprint import (
    ExcitationConditionedFingerprintOperator,
    FingerprintFactorEncoder,
)
from cvsrffi.phase1_fcr_nuisance import StructuredNuisanceEncoder
from cvsrffi.phase1_fcr_types import (
    FCRAggregateOutput,
    FCRConfig,
    FCRFactorOutput,
)

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


class NuisanceHeteroscedasticHead(nn.Module):
    """Predict standardized channel coordinates and their aleatoric scale."""

    def __init__(self, embedding_dim: int, nuisance_dim: int, hidden: Optional[int] = None):
        super().__init__()
        embedding_dim = int(embedding_dim)
        nuisance_dim = int(nuisance_dim)
        if embedding_dim < 1 or nuisance_dim < 1:
            raise ValueError("embedding_dim and nuisance_dim must be positive")
        hidden_dim = int(hidden or max(32, embedding_dim // 2))
        self.nuisance_dim = nuisance_dim
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * nuisance_dim),
        )

    def forward(self, z_dom: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_variance = self.net(z_dom).chunk(2, dim=-1)
        return mean, log_variance.clamp(-8.0, 8.0)


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
    physical_gate_variant: str = "none",
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
            physical_gate_variant=str(physical_gate_variant),
        )
    if str(physical_gate_variant or "none").lower().strip() != "none":
        raise ValueError("physical_gate_variant is supported only by cvsincnet")
    return FeatureBackboneAdapter(
        family,
        num_classes=int(num_classes),
        input_len=int(input_len),
        sample_rate_hz=float(sample_rate_hz),
        drop=0.0,
    )


FCR_FEATURE_SCHEMA = "ADV3B02:FCR:z_f_id:unit_l2:160:v1"


class ADV3B02FactorizedCrossReconstruction(nn.Module):
    """Compose the committed FCR factors without adding a waveform bypass."""

    def __init__(self, config: FCRConfig) -> None:
        super().__init__()
        self.config = config
        self.canonicalizer = ConservativeCanonicalizer(config)
        self.content = ContentFactorEncoder(config)
        self.fingerprint = FingerprintFactorEncoder(config)
        self.fingerprint_operator = ExcitationConditionedFingerprintOperator(config)
        self.nuisance = StructuredNuisanceEncoder(config)
        self.decoder = PhysicsOrderedDecoder(config)

    def forward(
        self,
        x: torch.Tensor,
        id_feature_raw: torch.Tensor,
        *,
        pair_context=None,
    ) -> FCRAggregateOutput:
        # CUDA ComplexHalf has incomplete operator coverage (for example,
        # gather). Keep the compact FCR physics branch in FP32/complex64 while
        # allowing the surrounding ADV3B02 backbone to remain under AMP.
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self._forward_fp32(
                x.float(), id_feature_raw.float(), pair_context=pair_context
            )

    def _forward_fp32(
        self,
        x: torch.Tensor,
        id_feature_raw: torch.Tensor,
        *,
        pair_context=None,
    ) -> FCRAggregateOutput:
        # Pair context is intentionally optional. Task10 may consume it for
        # paired losses; single-view factorization never requires a companion.
        del pair_context
        canonical = self.canonicalizer(x)
        content = self.content(canonical.canonical_iq)
        fingerprint_excitation = content.s_hat.detach()
        fingerprint = self.fingerprint(
            id_feature_raw,
            canonical.canonical_iq,
            canonical.residual_iq,
            excitation_features(fingerprint_excitation),
        )
        response = self.fingerprint_operator(fingerprint_excitation, fingerprint)
        nuisance = self.nuisance(x, canonical.eta_hat)
        decoded = self.decoder(content.s_hat, response.delta_f, nuisance)
        z_n_parts = {
            "channel": nuisance.z_ch,
            "receiver": nuisance.z_rx,
            "sync": nuisance.z_sync,
            "gain": nuisance.z_gain,
            "eta_pred": nuisance.eta_pred,
        }
        factors = FCRFactorOutput(
            z_s=content.z_s,
            z_f_id=fingerprint.z_f_id,
            z_tx_state=fingerprint.z_tx_state,
            z_n_parts=z_n_parts,
            s_hat=content.s_hat,
            content_confidence=content.content_confidence,
            response_coef=response.response_coef,
            response_quality=response.response_quality,
        )
        nuisance_vector = torch.cat(
            (nuisance.z_ch, nuisance.z_rx, nuisance.z_sync, nuisance.z_gain), dim=1
        )
        quality = {
            **{
                f"canonical_{name}": value
                for name, value in canonical.quality.items()
            },
            "content_confidence": content.content_confidence,
            **{
                f"fingerprint_{name}": value
                for name, value in response.response_quality.items()
            },
            "nuisance_norm": nuisance_vector.norm(dim=1),
            "decode_variance_mean": decoded.log_variance.exp().mean(dim=1),
        }
        return FCRAggregateOutput(
            canonical=canonical,
            content=content,
            fingerprint=fingerprint,
            response=response,
            nuisance=nuisance,
            factors=factors,
            decode=decoded,
            quality=quality,
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
        physical_gate_variant: str = "none",
        use_fcr: bool = False,
        fcr_config: Optional[FCRConfig] = None,
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
        self.use_fcr = bool(use_fcr)
        self.fcr_config = None
        self.fcr = None
        self.fcr_identity_head = None
        self.physical_gate_variant = str(
            physical_gate_variant or "none"
        ).lower().strip()
        if self.physical_gate_variant not in {"none", "nmfdu_v1"}:
            raise ValueError(
                "physical_gate_variant must be one of: none,nmfdu_v1"
            )
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
            physical_gate_variant=self.physical_gate_variant,
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
            physical_gate_variant="none",
        )
        if self.arch_family == "cvsincnet" and self.model_variant in {"lite_b", "lite_d", "lite_e", "lite_f", "lite_g", "lite_h"}:
            self._share_early_stem()

        self.emb_dim = self._infer_emb_dim(self.id_backbone)
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

        if self.use_fcr:
            self.fcr_config = (
                fcr_config if fcr_config is not None else FCRConfig(input_len=int(input_len))
            )
            self.fcr = ADV3B02FactorizedCrossReconstruction(self.fcr_config)
            self.fcr_identity_head = nn.Linear(self.emb_dim, self.num_classes)

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

    def _attach_fcr_outputs(
        self,
        out: Dict[str, object],
        x: torch.Tensor,
        z_id: torch.Tensor,
    ) -> Dict[str, object]:
        if not self.use_fcr:
            return out
        if self.fcr is None:
            raise RuntimeError("use_fcr=True requires an instantiated FCR module")
        if self.fcr_identity_head is None:
            raise RuntimeError("use_fcr=True requires an explicit FCR identity head")
        aggregate = self.fcr(x, z_id)
        fcr_tx_logits = self.fcr_identity_head(aggregate.factors.z_f_id)
        out.update(
            {
                "z_id_raw": z_id,
                "z_f_id": aggregate.factors.z_f_id,
                "z_tx_state": aggregate.factors.z_tx_state,
                "z_s": aggregate.factors.z_s,
                "z_n": {
                    name: aggregate.factors.z_n_parts[name]
                    for name in ("channel", "receiver", "sync", "gain")
                },
                "fcr_tx_logits": fcr_tx_logits,
                "fcr_decode": aggregate.decode,
                "fcr_decoder_mode": aggregate.decode.decoder_mode,
                "fcr_quality": aggregate.quality,
                "feature_schema": FCR_FEATURE_SCHEMA,
            }
        )
        return out

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
            out = {
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
            return self._attach_fcr_outputs(out, x, z_id)

        if (
            (not return_aux)
            and self.fast_infer_when_no_aux
            and self.sat_anchor_identity_adapter is None
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
        return self._attach_fcr_outputs(out, x, z_id)


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
    physical_gate_variant: str = "none",
    use_fcr: bool = False,
    fcr_config: Optional[FCRConfig] = None,
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
        physical_gate_variant=physical_gate_variant,
        use_fcr=use_fcr,
        fcr_config=fcr_config,
    )
