
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

try:
    from model_modified import build_model as build_single_model
except Exception:
    from model import build_model as build_single_model


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
        ):
            fallback.pop(key, None)
        return build_single_model(**fallback)


def backbone_forward_compat(backbone, x, *, y=None, return_aux: bool = True, domain_labels=None):
    try:
        return backbone(x, y=y, return_aux=return_aux, domain_labels=domain_labels)
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


class RCNStatEncoder(nn.Module):
    """Lightweight receiver/channel/noise statistic encoder for the domain path."""

    stat_dim = 18

    def __init__(self, out_dim: int, hidden: Optional[int] = None, drop: float = 0.05, eps: float = 1e-6):
        super().__init__()
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

        return torch.stack(
            [
                i_mean, i_std, i_abs,
                q_mean, q_std, q_abs,
                a_mean, a_std, a_abs,
                p_mean, p_std,
                iq_corr.clamp(-5.0, 5.0), iq_imbalance.clamp(-5.0, 5.0),
                di_abs, dq_abs, da_abs,
                dphi_mean.clamp(-math.pi, math.pi), dphi_std + 0.1 * dphi_abs,
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
        if self.mode not in ("off", "rcn_stats"):
            raise ValueError("domain_enhancer must be one of: off,rcn_stats")
        if self.mode == "off" or self.strength <= 0.0:
            self.stat_encoder = None
            self.gate = None
            self.norm = nn.Identity()
        else:
            self.stat_encoder = RCNStatEncoder(emb_dim, hidden=max(64, emb_dim // 2), drop=drop)
            self.gate = nn.Sequential(nn.Linear(2 * emb_dim, emb_dim), nn.Sigmoid())
            self.norm = nn.LayerNorm(emb_dim)

    def forward(self, z_dom: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.stat_encoder is None or self.gate is None:
            return z_dom, None
        z_rcn = self.stat_encoder(x)
        gate = self.gate(torch.cat([z_dom, z_rcn], dim=1))
        return self.norm(z_dom + self.strength * gate * z_rcn), z_rcn


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
        fast_infer_when_no_aux: bool = True,
        sgc_adapter: bool = False,
        sgc_adapter_kwargs: Optional[Dict] = None,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_domains = int(max(1, num_domains))
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
        if bool(sgc_adapter):
            from sgc_adapter import SGCAdapter

            self.sgc_adapter = SGCAdapter(**(sgc_adapter_kwargs or {}))
        else:
            self.sgc_adapter = None

        self.id_backbone = build_single_model_compat(
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
        )
        self.dom_backbone = build_single_model_compat(
            num_classes=num_classes,
            model_size=model_size,
            dataset=dataset,
            input_len=input_len,
            sample_rate_hz=sample_rate_hz,
            model_variant=self.model_variant,
            branch_ablation=self.domain_branch_ablation,
            mixstyle_on=False,
        )
        if self.model_variant in {"lite_b", "lite_d", "lite_e"}:
            self._share_early_stem()

        self.emb_dim = self._infer_emb_dim(self.id_backbone)
        self.dom_enhancer = DomainFeatureEnhancer(
            self.emb_dim,
            mode=str(domain_enhancer),
            strength=float(domain_enhancer_strength),
            drop=drop * 0.5,
        )
        self.dom_head = MLPHead(self.emb_dim, self.num_domains, hidden=max(64, self.emb_dim // 2), drop=drop)
        self.adv_head = MLPHead(self.emb_dim, self.num_domains, hidden=max(64, self.emb_dim // 2), drop=drop)

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

    def forward(
        self,
        x: torch.Tensor,
        y_tx: Optional[torch.Tensor] = None,
        grl_lambda: float = 1.0,
        return_aux: bool = False,
        domain_labels: Optional[torch.Tensor] = None,
    ):
        if self.sgc_adapter is not None:
            x_processed, sgc_aux = self.sgc_adapter(x, return_aux=True)
        else:
            x_processed = x
            sgc_aux = {}

        if (not return_aux) and self.fast_infer_when_no_aux:
            return backbone_forward_compat(
                self.id_backbone,
                x_processed,
                y=y_tx,
                return_aux=False,
                domain_labels=domain_labels,
            )

        aux_id = backbone_forward_compat(self.id_backbone, x_processed, y=y_tx, return_aux=True, domain_labels=domain_labels)
        aux_dom = backbone_forward_compat(self.dom_backbone, x_processed, y=None, return_aux=True, domain_labels=None)

        tx_logits = aux_id["logits"]
        z_id = self._pick_z_id(aux_id)
        z_dom_raw = self._pick_z_dom(aux_dom)
        z_dom, z_dom_rcn = self.dom_enhancer(z_dom_raw, x_processed)

        dom_logits = self.dom_head(z_dom)
        adv_dom_logits = self.adv_head(grad_reverse(z_id, grl_lambda))

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
            "aux_id": aux_id,
            "aux_dom": aux_dom,
        }
        out["sgc_aux"] = sgc_aux
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
    fast_infer_when_no_aux: bool = True,
    sgc_adapter: bool = False,
    sgc_adapter_kwargs: Optional[Dict] = None,
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
        fast_infer_when_no_aux=fast_infer_when_no_aux,
        sgc_adapter=sgc_adapter,
        sgc_adapter_kwargs=sgc_adapter_kwargs,
    )
