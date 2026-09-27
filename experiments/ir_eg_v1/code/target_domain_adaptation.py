from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TargetAdaptLossConfig:
    entropy_weight: float = 1.0
    consistency_weight: float = 0.5
    pseudo_weight: float = 0.5
    anchor_weight: float = 0.05
    conf_threshold: float = 0.90
    margin_threshold: float = 0.20
    anchor_temperature: float = 2.0


class EntropyMinimizationLogitAdapter(nn.Module):
    """Source-free target adaptation wrapper for a trained classifier.

    The base model can remain mostly frozen while the wrapper learns a small
    class-wise logit calibration. Optional norm/classifier unfreezing is handled
    by configure_target_adaptation_parameters.
    """

    def __init__(self, base_model: nn.Module, num_classes: int, *, freeze_base_stats: bool = False) -> None:
        super().__init__()
        self.base_model = base_model
        self.logit_bias = nn.Parameter(torch.zeros(int(num_classes)))
        self.log_temperature = nn.Parameter(torch.zeros(()))
        self.freeze_base_stats = bool(freeze_base_stats)
        self.adapter_type = "logit_calibration"

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_base_stats:
            self.base_model.eval()
        return self

    def forward(self, x, y_tx=None, grl_lambda: float = 1.0, return_aux: bool = True, domain_labels=None):
        if self.freeze_base_stats:
            self.base_model.eval()
        out = self.base_model(
            x,
            y_tx=y_tx,
            grl_lambda=grl_lambda,
            return_aux=True,
            domain_labels=domain_labels,
        )
        if torch.is_tensor(out):
            out_dict: Dict[str, Any] = {"tx_logits": out}
        else:
            out_dict = dict(out)
        logits = out_dict["tx_logits"]
        temperature = self.log_temperature.exp().clamp(0.05, 20.0)
        out_dict["base_tx_logits"] = logits.detach()
        out_dict["tx_logits"] = logits / temperature + self.logit_bias.view(1, -1)
        if "dom_logits" not in out_dict or not torch.is_tensor(out_dict["dom_logits"]):
            out_dict["dom_logits"] = out_dict["tx_logits"].new_zeros(out_dict["tx_logits"].size(0), 1)
        return out_dict

    def adaptation_parameters(self) -> Iterable[nn.Parameter]:
        return [self.logit_bias, self.log_temperature]

    def adapter_state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "logit_bias": self.logit_bias.detach().clone(),
            "log_temperature": self.log_temperature.detach().clone(),
        }


class OnOrbitLogitLoRAAdapter(nn.Module):
    """Low-rank logit residual adapter for frozen on-orbit updates."""

    def __init__(
        self,
        base_model: nn.Module,
        num_classes: int,
        *,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
        freeze_base_stats: bool = True,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.num_classes = int(num_classes)
        self.rank = max(1, int(rank))
        self.alpha = float(alpha)
        self.freeze_base_stats = bool(freeze_base_stats)
        self.dropout = nn.Dropout(float(dropout))
        self.lora_a = nn.Linear(self.num_classes, self.rank, bias=False)
        self.lora_b = nn.Linear(self.rank, self.num_classes, bias=False)
        self.logit_bias = nn.Parameter(torch.zeros(self.num_classes))
        self.log_temperature = nn.Parameter(torch.zeros(()))
        self.adapter_type = "logit_lora"
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_model.eval()
        return self

    def _base_forward(self, x, y_tx=None, grl_lambda: float = 1.0, domain_labels=None) -> Dict[str, Any]:
        self.base_model.eval()
        with torch.no_grad():
            out = self.base_model(
                x,
                y_tx=y_tx,
                grl_lambda=grl_lambda,
                return_aux=True,
                domain_labels=domain_labels,
            )
        return {"tx_logits": out} if torch.is_tensor(out) else dict(out)

    def forward(self, x, y_tx=None, grl_lambda: float = 1.0, return_aux: bool = True, domain_labels=None):
        out_dict = self._base_forward(x, y_tx=y_tx, grl_lambda=grl_lambda, domain_labels=domain_labels)
        base_logits = out_dict["tx_logits"].detach()
        temperature = self.log_temperature.exp().clamp(0.05, 20.0)
        delta = self.lora_b(self.lora_a(self.dropout(base_logits.float()))) * (self.alpha / float(self.rank))
        out_dict["base_tx_logits"] = base_logits
        out_dict["tx_logits"] = base_logits / temperature + self.logit_bias.view(1, -1) + delta.to(dtype=base_logits.dtype)
        out_dict["adapter_delta_logits"] = delta
        out_dict["adapter_delta_logit_norm"] = delta.float().norm(dim=-1)
        if "dom_logits" not in out_dict or not torch.is_tensor(out_dict["dom_logits"]):
            out_dict["dom_logits"] = out_dict["tx_logits"].new_zeros(out_dict["tx_logits"].size(0), 1)
        return out_dict

    def adaptation_parameters(self) -> Iterable[nn.Parameter]:
        return [self.logit_bias, self.log_temperature, *self.lora_a.parameters(), *self.lora_b.parameters()]

    def adapter_state_dict(self) -> Dict[str, torch.Tensor]:
        keys = ["logit_bias", "log_temperature", "lora_a.weight", "lora_b.weight"]
        return {key: self.state_dict()[key] for key in keys}


class OnOrbitFeatureResidualAdapter(nn.Module):
    """Frozen-backbone feature adapter that learns a small feature-to-logit delta.

    The base model must expose `z_id` in its aux output. The residual head is
    zero-initialized at the final layer, so the first forward is exactly the
    frozen base prediction until adaptation updates the small head.
    """

    def __init__(
        self,
        base_model: nn.Module,
        num_classes: int,
        *,
        bottleneck: int = 16,
        alpha: float = 1.0,
        dropout: float = 0.0,
        freeze_base_stats: bool = True,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.num_classes = int(num_classes)
        self.bottleneck = max(1, int(bottleneck))
        self.alpha = float(alpha)
        self.freeze_base_stats = bool(freeze_base_stats)
        self.adapter_down = nn.LazyLinear(self.bottleneck)
        self.adapter_up = nn.Linear(self.bottleneck, self.num_classes)
        self.logit_bias = nn.Parameter(torch.zeros(self.num_classes))
        self.log_temperature = nn.Parameter(torch.zeros(()))
        self.dropout = nn.Dropout(float(dropout))
        self.adapter_type = "feature_residual"
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_model.eval()
        return self

    def _base_forward(self, x, y_tx=None, grl_lambda: float = 1.0, domain_labels=None) -> Dict[str, Any]:
        self.base_model.eval()
        with torch.no_grad():
            out = self.base_model(
                x,
                y_tx=y_tx,
                grl_lambda=grl_lambda,
                return_aux=True,
                domain_labels=domain_labels,
            )
        return {"tx_logits": out} if torch.is_tensor(out) else dict(out)

    def forward(self, x, y_tx=None, grl_lambda: float = 1.0, return_aux: bool = True, domain_labels=None):
        out_dict = self._base_forward(x, y_tx=y_tx, grl_lambda=grl_lambda, domain_labels=domain_labels)
        if "z_id" not in out_dict:
            raise KeyError("feature_residual adapter requires base model aux output key 'z_id'")
        base_logits = out_dict["tx_logits"].detach()
        z = out_dict["z_id"].detach().float()
        z_norm = F.layer_norm(z, z.shape[-1:])
        hidden = F.gelu(self.adapter_down(z_norm))
        delta = self.adapter_up(self.dropout(hidden)) * self.alpha
        temperature = self.log_temperature.exp().clamp(0.05, 20.0)
        out_dict["base_tx_logits"] = base_logits
        out_dict["tx_logits"] = base_logits / temperature + self.logit_bias.view(1, -1) + delta.to(dtype=base_logits.dtype)
        out_dict["z_base"] = z
        out_dict["adapter_delta_logits"] = delta
        out_dict["adapter_delta_logit_norm"] = delta.float().norm(dim=-1)
        if "dom_logits" not in out_dict or not torch.is_tensor(out_dict["dom_logits"]):
            out_dict["dom_logits"] = out_dict["tx_logits"].new_zeros(out_dict["tx_logits"].size(0), 1)
        return out_dict

    def adaptation_parameters(self) -> Iterable[nn.Parameter]:
        return [
            self.logit_bias,
            self.log_temperature,
            *self.adapter_down.parameters(),
            *self.adapter_up.parameters(),
        ]

    def adapter_state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            key: value
            for key, value in self.state_dict().items()
            if key.startswith(("logit_", "adapter_down.", "adapter_up."))
        }


def build_target_adapter(
    base_model: nn.Module,
    *,
    num_classes: int,
    adapter_type: str = "logit_calibration",
    adapter_rank: int = 4,
    adapter_bottleneck: int = 16,
    adapter_alpha: float = 1.0,
    adapter_dropout: float = 0.0,
    freeze_base_stats: bool = False,
) -> nn.Module:
    kind = str(adapter_type).lower()
    if kind in {"logit", "logit_calibration", "calibration"}:
        return EntropyMinimizationLogitAdapter(
            base_model,
            num_classes=int(num_classes),
            freeze_base_stats=bool(freeze_base_stats),
        )
    if kind in {"logit_lora", "lora", "lora_head"}:
        return OnOrbitLogitLoRAAdapter(
            base_model,
            num_classes=int(num_classes),
            rank=int(adapter_rank),
            alpha=float(adapter_alpha),
            dropout=float(adapter_dropout),
            freeze_base_stats=True,
        )
    if kind in {"feature_residual", "feature_adapter", "sgc_adapter"}:
        return OnOrbitFeatureResidualAdapter(
            base_model,
            num_classes=int(num_classes),
            bottleneck=int(adapter_bottleneck),
            alpha=float(adapter_alpha),
            dropout=float(adapter_dropout),
            freeze_base_stats=True,
        )
    raise ValueError(f"unknown target adapter type: {adapter_type}")


def configure_target_adaptation_parameters(
    model: nn.Module,
    *,
    update_norm: bool = True,
    update_classifier: bool = False,
) -> List[nn.Parameter]:
    for param in model.parameters():
        param.requires_grad = False

    if hasattr(model, "adaptation_parameters"):
        for param in model.adaptation_parameters():
            param.requires_grad = True
    else:
        model.logit_bias.requires_grad = True
        model.log_temperature.requires_grad = True

    norm_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm, nn.InstanceNorm1d, nn.InstanceNorm2d)
    base_model = getattr(model, "base_model", None)
    if update_norm and base_model is not None and str(getattr(model, "adapter_type", "")) == "logit_calibration":
        for module in base_model.modules():
            if isinstance(module, norm_types):
                for param in module.parameters(recurse=False):
                    param.requires_grad = True

    if update_classifier and base_model is not None and str(getattr(model, "adapter_type", "")) == "logit_calibration":
        for name, module in base_model.named_modules():
            lname = name.lower()
            if isinstance(module, nn.Linear) and any(token in lname for token in ("head", "classifier", "fc", "logit")):
                for param in module.parameters(recurse=False):
                    param.requires_grad = True

    return [param for param in model.parameters() if param.requires_grad]


def probability_margin(prob: torch.Tensor) -> torch.Tensor:
    top2 = prob.topk(min(2, prob.size(-1)), dim=-1).values
    if top2.size(-1) == 1:
        return top2[:, 0]
    return top2[:, 0] - top2[:, 1]


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    log_prob = F.log_softmax(logits.float(), dim=-1)
    prob = log_prob.exp()
    return -(prob * log_prob).sum(dim=-1)


def symmetric_kl(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    log_pa = F.log_softmax(logits_a.float(), dim=-1)
    log_pb = F.log_softmax(logits_b.float(), dim=-1)
    pa = log_pa.exp().detach()
    pb = log_pb.exp().detach()
    return 0.5 * (
        F.kl_div(log_pa, pb, reduction="batchmean")
        + F.kl_div(log_pb, pa, reduction="batchmean")
    )


def _anchor_kl(adapted_logits: torch.Tensor, base_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    temp = max(1e-6, float(temperature))
    log_q = F.log_softmax(adapted_logits.float() / temp, dim=-1)
    p = F.softmax(base_logits.detach().float() / temp, dim=-1)
    return F.kl_div(log_q, p, reduction="batchmean") * (temp * temp)


def compute_target_adaptation_loss(
    model: nn.Module,
    x_clean: torch.Tensor,
    x_sat: torch.Tensor | None,
    cfg: TargetAdaptLossConfig | None = None,
):
    cfg = cfg or TargetAdaptLossConfig()
    out_clean = model(x_clean)
    logits_clean = out_clean["tx_logits"]
    out_sat = model(x_sat) if x_sat is not None else None
    logits_sat = out_sat["tx_logits"] if out_sat is not None else None
    logits_mean = 0.5 * (logits_clean + logits_sat) if logits_sat is not None else logits_clean
    prob_mean = F.softmax(logits_mean.detach().float(), dim=-1)
    conf, pseudo_y = prob_mean.max(dim=-1)
    margin = probability_margin(prob_mean)
    pseudo_mask = (conf >= float(cfg.conf_threshold)) & (margin >= float(cfg.margin_threshold))

    entropy_loss = entropy_from_logits(logits_mean).mean()
    consistency_loss = symmetric_kl(logits_clean, logits_sat) if logits_sat is not None else logits_mean.sum() * 0.0
    if pseudo_mask.any():
        ce_clean = F.cross_entropy(logits_clean, pseudo_y, reduction="none")
        if logits_sat is not None:
            ce_sat = F.cross_entropy(logits_sat, pseudo_y, reduction="none")
            pseudo_loss = 0.5 * (ce_clean[pseudo_mask].mean() + ce_sat[pseudo_mask].mean())
        else:
            pseudo_loss = ce_clean[pseudo_mask].mean()
    else:
        pseudo_loss = logits_mean.sum() * 0.0

    anchor_loss = logits_mean.sum() * 0.0
    if "base_tx_logits" in out_clean:
        anchor_loss = anchor_loss + _anchor_kl(logits_clean, out_clean["base_tx_logits"], cfg.anchor_temperature)
    if out_sat is not None and "base_tx_logits" in out_sat:
        anchor_loss = anchor_loss + _anchor_kl(logits_sat, out_sat["base_tx_logits"], cfg.anchor_temperature)
        anchor_loss = 0.5 * anchor_loss

    loss = (
        float(cfg.entropy_weight) * entropy_loss
        + float(cfg.consistency_weight) * consistency_loss
        + float(cfg.pseudo_weight) * pseudo_loss
        + float(cfg.anchor_weight) * anchor_loss
    )
    logs = {
        "target_adapt/loss_total": loss.detach(),
        "target_adapt/loss_entropy": entropy_loss.detach(),
        "target_adapt/loss_consistency": consistency_loss.detach(),
        "target_adapt/loss_pseudo": pseudo_loss.detach(),
        "target_adapt/loss_anchor": anchor_loss.detach(),
        "target_adapt/pseudo_coverage": pseudo_mask.float().mean().detach(),
        "target_adapt/pseudo_confidence": conf.mean().detach(),
        "target_adapt/pseudo_margin": margin.mean().detach(),
    }
    return loss, logs


def _label_at(dataset, idx: int) -> int:
    if hasattr(dataset, "index"):
        item = dataset.index[int(idx)]
        if hasattr(item, "tx_i"):
            return int(item.tx_i)
    if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
        inner_idx = int(dataset.indices[int(idx)])
        return _label_at(dataset.dataset, inner_idx)
    sample = dataset[int(idx)]
    if isinstance(sample, (tuple, list)) and len(sample) >= 2:
        return int(sample[1])
    raise ValueError("dataset samples must expose labels as the second tuple item")


def select_fewshot_indices(
    dataset,
    *,
    samples_per_class: int = 0,
    max_samples: int = 0,
    seed: int = 0,
) -> List[int]:
    total = len(dataset)
    if total <= 0:
        return []
    gen = torch.Generator()
    gen.manual_seed(int(seed))

    if int(samples_per_class) > 0:
        by_class: Dict[int, List[int]] = {}
        for idx in range(total):
            by_class.setdefault(_label_at(dataset, idx), []).append(idx)
        selected: List[int] = []
        for label in sorted(by_class):
            indices = by_class[label]
            order = torch.randperm(len(indices), generator=gen).tolist()
            take = min(int(samples_per_class), len(indices))
            selected.extend(indices[i] for i in order[:take])
        order = torch.randperm(len(selected), generator=gen).tolist()
        selected = [selected[i] for i in order]
    else:
        selected = torch.randperm(total, generator=gen).tolist()

    if int(max_samples) > 0:
        selected = selected[: int(max_samples)]
    return selected


def select_unlabeled_target_indices(dataset, *, num_samples: int = 0, seed: int = 0) -> List[int]:
    total = len(dataset)
    if total <= 0:
        return []
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    selected = torch.randperm(total, generator=gen).tolist()
    if int(num_samples) > 0:
        selected = selected[: min(int(num_samples), total)]
    return selected


def _rx_at(dataset, idx: int) -> Optional[int]:
    if hasattr(dataset, "index"):
        item = dataset.index[int(idx)]
        if hasattr(item, "rx_i"):
            return int(item.rx_i)
        if isinstance(item, Mapping) and "rx_i" in item:
            return int(item["rx_i"])
    if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
        inner_idx = int(dataset.indices[int(idx)])
        return _rx_at(dataset.dataset, inner_idx)
    sample = dataset[int(idx)]
    if isinstance(sample, (tuple, list)) and len(sample) >= 4 and isinstance(sample[3], Mapping):
        meta = sample[3]
        if "rx_i" in meta:
            return int(meta["rx_i"])
    return None


def select_unlabeled_target_indices_per_rx(dataset, *, samples_per_rx: int = 0, seed: int = 0) -> List[int]:
    total = len(dataset)
    if total <= 0 or int(samples_per_rx) <= 0:
        return []
    by_rx: Dict[int, List[int]] = defaultdict(list)
    for idx in range(total):
        rx = _rx_at(dataset, idx)
        if rx is None:
            return select_unlabeled_target_indices(dataset, num_samples=int(samples_per_rx), seed=seed)
        by_rx[int(rx)].append(idx)

    gen = torch.Generator()
    gen.manual_seed(int(seed))
    selected: List[int] = []
    for rx in sorted(by_rx):
        indices = by_rx[rx]
        order = torch.randperm(len(indices), generator=gen).tolist()
        take = min(int(samples_per_rx), len(indices))
        selected.extend(indices[i] for i in order[:take])
    if selected:
        order = torch.randperm(len(selected), generator=gen).tolist()
        selected = [selected[i] for i in order]
    return selected


def select_target_indices_per_rx_tx(dataset, *, samples_per_rx_tx: int = 0, seed: int = 0) -> List[int]:
    total = len(dataset)
    if total <= 0 or int(samples_per_rx_tx) <= 0:
        return []
    by_rx_tx: Dict[tuple[int, int], List[int]] = defaultdict(list)
    for idx in range(total):
        rx = _rx_at(dataset, idx)
        if rx is None:
            return select_fewshot_indices(dataset, samples_per_class=int(samples_per_rx_tx), seed=seed)
        tx = _label_at(dataset, idx)
        by_rx_tx[(int(rx), int(tx))].append(idx)

    gen = torch.Generator()
    gen.manual_seed(int(seed))
    selected: List[int] = []
    for key in sorted(by_rx_tx):
        indices = by_rx_tx[key]
        order = torch.randperm(len(indices), generator=gen).tolist()
        take = min(int(samples_per_rx_tx), len(indices))
        selected.extend(indices[i] for i in order[:take])
    if selected:
        order = torch.randperm(len(selected), generator=gen).tolist()
        selected = [selected[i] for i in order]
    return selected
