from __future__ import annotations

import math
import random
import re
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F


_TX_META_KEYS = ("tx_id", "tx_i", "label", "y", "tx")


def _safe_l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
    return F.normalize(x, dim=dim, eps=float(eps))


def _clip_and_normalize(vec: torch.Tensor, clip_norm: float = 0.0) -> torch.Tensor:
    vec = torch.nan_to_num(vec.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if float(clip_norm) > 0.0:
        norm = torch.linalg.vector_norm(vec, ord=2).clamp_min(1e-6)
        vec = vec * min(1.0, float(clip_norm) / float(norm.item()))
    return _safe_l2_normalize(vec.view(1, -1), dim=1).squeeze(0)


def _metadata_value(meta: Mapping[str, Any], keys: Iterable[str]) -> Optional[Any]:
    for key in keys:
        if key in meta:
            return meta[key]
    return None


def _as_client_count(count: Any) -> float:
    try:
        value = float(count)
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) and value > 0.0 else 0.0


def client_domain_from_id(client_id: str) -> str:
    """Use receiver identity as the default VMB domain for receiver-client FL."""
    text = str(client_id)
    match = re.match(r"^(rx[^_]+)", text)
    if match:
        return match.group(1)
    return text


def domain_balanced_weights(
    client_ids: Sequence[str],
    domain_ids: Mapping[str, Any],
) -> Dict[str, float]:
    if not client_ids:
        raise ValueError("client_ids must not be empty")
    groups: Dict[str, list[str]] = defaultdict(list)
    for cid in client_ids:
        groups[str(domain_ids.get(cid, client_domain_from_id(str(cid))))].append(str(cid))
    domain_weight = 1.0 / float(max(1, len(groups)))
    weights: Dict[str, float] = {}
    for members in groups.values():
        per_client = domain_weight / float(max(1, len(members)))
        for cid in members:
            weights[cid] = per_client
    return weights


def select_domain_balanced_clients(
    client_ids: Sequence[str],
    domain_ids: Mapping[str, Any],
    *,
    clients_per_round: int,
    seed: int,
    round_idx: int,
) -> list[str]:
    ids = [str(cid) for cid in client_ids]
    if not ids:
        return []
    k = max(1, min(int(clients_per_round), len(ids)))
    rng = random.Random(int(seed) + int(round_idx) * 1009 + 17)
    groups: Dict[str, list[str]] = defaultdict(list)
    for cid in ids:
        groups[str(domain_ids.get(cid, client_domain_from_id(cid)))].append(cid)
    ordered_domains = sorted(groups.keys())
    rng.shuffle(ordered_domains)
    for members in groups.values():
        rng.shuffle(members)

    selected: list[str] = []
    while len(selected) < k and ordered_domains:
        progressed = False
        for domain in list(ordered_domains):
            members = groups.get(domain, [])
            if not members:
                continue
            selected.append(members.pop(0))
            progressed = True
            if len(selected) >= k:
                break
        ordered_domains = [d for d in ordered_domains if groups.get(d)]
        if not progressed:
            break
    if len(selected) < k:
        remaining = [cid for cid in ids if cid not in selected]
        rng.shuffle(remaining)
        selected.extend(remaining[: k - len(selected)])
    return sorted(selected)


def select_transmitter_balanced_indices(
    indices: Sequence[int],
    metadata_fn: Callable[[int], Mapping[str, Any]],
    *,
    batch_size: int,
    seed: int,
    round_idx: int,
    batch_idx: int,
) -> list[int]:
    ids = [int(idx) for idx in indices]
    if not ids:
        return []
    rng = random.Random(int(seed) + int(round_idx) * 1000003 + int(batch_idx) * 9176 + 31)
    groups: Dict[str, list[int]] = defaultdict(list)
    for idx in ids:
        try:
            meta = metadata_fn(int(idx)) or {}
            tx = _metadata_value(meta, _TX_META_KEYS)
        except Exception:
            tx = None
        groups[str(tx) if tx is not None else "__unknown__"].append(int(idx))
    for members in groups.values():
        rng.shuffle(members)
    tx_keys = sorted(groups.keys())
    rng.shuffle(tx_keys)

    selected: list[int] = []
    k = max(1, min(int(batch_size), len(ids)))
    while len(selected) < k and tx_keys:
        progressed = False
        for key in list(tx_keys):
            members = groups.get(key, [])
            if not members:
                continue
            selected.append(members.pop(0))
            progressed = True
            if len(selected) >= k:
                break
        tx_keys = [key for key in tx_keys if groups.get(key)]
        if not progressed:
            break
    if len(selected) < k:
        remaining = [idx for idx in ids if idx not in selected]
        rng.shuffle(remaining)
        selected.extend(remaining[: k - len(selected)])
    return selected


@dataclass
class FedCVSVMBPrototypeStats:
    tx_sum: torch.Tensor
    tx_count: torch.Tensor
    rx_sum_by_client: Mapping[str, torch.Tensor]
    rx_count_by_client: Mapping[str, float]


@dataclass
class FedCVSCoralStats:
    class_sum: torch.Tensor
    class_outer: torch.Tensor
    class_count: torch.Tensor
    mode: str = "diag"


def _safe_coral_mode(mode: str) -> str:
    text = str(mode or "diag").lower().strip()
    if text in {"diagonal", "var", "variance"}:
        return "diag"
    if text in {"full", "shrink", "full_shrink", "shrinkage"}:
        return "full"
    if text not in {"diag", "full"}:
        raise ValueError("fed CORAL mode must be 'diag' or 'full'")
    return text


def build_class_conditional_coral_stats(
    features: Optional[torch.Tensor],
    labels: torch.Tensor,
    *,
    num_classes: int,
    mode: str = "diag",
) -> Optional[FedCVSCoralStats]:
    if not torch.is_tensor(features) or features.dim() != 2 or int(features.size(0)) == 0:
        return None
    y = labels.detach().view(-1).cpu().long()
    z = torch.nan_to_num(features.detach().cpu().float(), nan=0.0, posinf=0.0, neginf=0.0)
    if int(y.numel()) != int(z.size(0)):
        return None
    c = max(1, int(num_classes))
    dim = int(z.size(1))
    coral_mode = _safe_coral_mode(mode)
    class_sum = torch.zeros(c, dim, dtype=torch.float32)
    if coral_mode == "diag":
        class_outer = torch.zeros(c, dim, dtype=torch.float32)
    else:
        class_outer = torch.zeros(c, dim, dim, dtype=torch.float32)
    class_count = torch.zeros(c, dtype=torch.float32)
    valid = (y >= 0) & (y < c)
    for cls in torch.unique(y[valid]):
        cls_i = int(cls.item())
        mask = valid & (y == cls)
        z_cls = z[mask]
        if int(z_cls.size(0)) == 0:
            continue
        class_sum[cls_i] = z_cls.sum(dim=0)
        class_count[cls_i] = float(z_cls.size(0))
        if coral_mode == "diag":
            class_outer[cls_i] = (z_cls * z_cls).sum(dim=0)
        else:
            class_outer[cls_i] = z_cls.t().mm(z_cls)
    return FedCVSCoralStats(class_sum=class_sum, class_outer=class_outer, class_count=class_count, mode=coral_mode)


def merge_coral_stats(stats_list: Iterable[Optional[FedCVSCoralStats]]) -> Optional[FedCVSCoralStats]:
    class_sum = None
    class_outer = None
    class_count = None
    mode = None
    for stats in stats_list:
        if stats is None:
            continue
        current_mode = _safe_coral_mode(stats.mode)
        if mode is None:
            mode = current_mode
        if current_mode != mode:
            raise ValueError(f"Cannot merge CORAL stats with different modes: {mode} vs {current_mode}")
        class_sum = (
            stats.class_sum.detach().cpu().float().clone()
            if class_sum is None
            else class_sum + stats.class_sum.detach().cpu().float()
        )
        class_outer = (
            stats.class_outer.detach().cpu().float().clone()
            if class_outer is None
            else class_outer + stats.class_outer.detach().cpu().float()
        )
        class_count = (
            stats.class_count.detach().cpu().float().clone()
            if class_count is None
            else class_count + stats.class_count.detach().cpu().float()
        )
    if class_sum is None or class_outer is None or class_count is None:
        return None
    return FedCVSCoralStats(class_sum=class_sum, class_outer=class_outer, class_count=class_count, mode=str(mode or "diag"))


class FedCVSCoralStatsBank:
    """Server-side class-conditional feature statistics for opt-in CORAL alignment."""

    def __init__(self, num_classes: int, momentum: float = 0.95, mode: str = "diag"):
        self.num_classes = max(1, int(num_classes))
        self.momentum = max(0.0, min(0.9999, float(momentum)))
        self.mode = _safe_coral_mode(mode)
        self.stats: Optional[FedCVSCoralStats] = None
        self.rounds_updated = 0

    def update(self, stats: Optional[FedCVSCoralStats]) -> Dict[str, Any]:
        if stats is None:
            return self.summary()
        incoming = FedCVSCoralStats(
            class_sum=stats.class_sum.detach().cpu().float().clone(),
            class_outer=stats.class_outer.detach().cpu().float().clone(),
            class_count=stats.class_count.detach().cpu().float().clone(),
            mode=_safe_coral_mode(stats.mode),
        )
        if incoming.mode != self.mode:
            raise ValueError(f"CORAL bank mode mismatch: bank={self.mode} incoming={incoming.mode}")
        if int(incoming.class_sum.size(0)) != self.num_classes:
            raise ValueError(f"CORAL num_classes mismatch: expected {self.num_classes}, got {int(incoming.class_sum.size(0))}")
        if self.stats is None or self.momentum <= 0.0:
            self.stats = incoming
        else:
            m = self.momentum
            self.stats = FedCVSCoralStats(
                class_sum=self.stats.class_sum * m + incoming.class_sum * (1.0 - m),
                class_outer=self.stats.class_outer * m + incoming.class_outer * (1.0 - m),
                class_count=self.stats.class_count * m + incoming.class_count * (1.0 - m),
                mode=self.mode,
            )
        self.rounds_updated += 1
        return self.summary()

    def summary(self) -> Dict[str, Any]:
        if self.stats is None:
            return {"enabled": True, "mode": self.mode, "rounds_updated": int(self.rounds_updated), "class_count_nonzero": 0}
        return {
            "enabled": True,
            "mode": self.mode,
            "rounds_updated": int(self.rounds_updated),
            "class_count_nonzero": int((self.stats.class_count > 0).sum().item()),
            "total_count": float(self.stats.class_count.sum().item()),
            "payload_bytes": coral_stats_payload_size_bytes(self.stats),
        }


def _coral_mean_and_cov(stats: FedCVSCoralStats, cls: int, *, device: torch.device, dtype: torch.dtype):
    count = stats.class_count[int(cls)].to(device=device, dtype=dtype).clamp_min(1.0)
    mean = stats.class_sum[int(cls)].to(device=device, dtype=dtype) / count
    if _safe_coral_mode(stats.mode) == "diag":
        second = stats.class_outer[int(cls)].to(device=device, dtype=dtype) / count
        cov = (second - mean * mean).clamp_min(0.0)
    else:
        second = stats.class_outer[int(cls)].to(device=device, dtype=dtype) / count
        cov = second - torch.outer(mean, mean)
    return mean, cov


def _shrink_full_covariance(cov: torch.Tensor, shrinkage: float) -> torch.Tensor:
    alpha = max(0.0, min(1.0, float(shrinkage)))
    if alpha <= 0.0 or cov.dim() != 2:
        return cov
    diag = torch.diag(torch.diagonal(cov))
    return cov * (1.0 - alpha) + diag * alpha


def class_conditional_coral_loss(
    features: Optional[torch.Tensor],
    labels: torch.Tensor,
    target_stats: Optional[FedCVSCoralStats],
    *,
    min_count: int = 2,
    mean_weight: float = 1.0,
    cov_weight: float = 1.0,
    shrinkage: float = 0.0,
) -> tuple[torch.Tensor, Dict[str, float]]:
    ref = features if torch.is_tensor(features) else torch.tensor(0.0)
    zero = ref.new_tensor(0.0)
    empty = {
        "active_classes": 0,
        "samples": 0,
        "mean_dist": float("nan"),
        "cov_dist": float("nan"),
        "skip_rate": 1.0,
    }
    if not torch.is_tensor(features) or features.dim() != 2 or target_stats is None:
        return zero, empty
    y = labels.to(device=features.device).view(-1).long()
    if int(y.numel()) != int(features.size(0)):
        return zero, empty
    mode = _safe_coral_mode(target_stats.mode)
    losses = []
    mean_dists = []
    cov_dists = []
    active_classes = 0
    total_classes = max(1, int(target_stats.class_count.numel()))
    for cls in range(total_classes):
        if float(target_stats.class_count[int(cls)].item()) < max(1, int(min_count)):
            continue
        mask = y == int(cls)
        if int(mask.sum().item()) < max(1, int(min_count)):
            continue
        z = torch.nan_to_num(features[mask].float(), nan=0.0, posinf=0.0, neginf=0.0)
        mean = z.mean(dim=0)
        if mode == "diag":
            cov = z.var(dim=0, unbiased=False)
        else:
            centered = z - mean
            cov = centered.t().mm(centered) / float(max(1, int(z.size(0)) - 1))
        target_mean, target_cov = _coral_mean_and_cov(target_stats, cls, device=features.device, dtype=z.dtype)
        if mode == "full":
            cov = _shrink_full_covariance(cov, shrinkage)
            target_cov = _shrink_full_covariance(target_cov, shrinkage)
        mean_loss = F.mse_loss(mean, target_mean)
        cov_loss = F.mse_loss(cov, target_cov)
        losses.append(float(mean_weight) * mean_loss + float(cov_weight) * cov_loss)
        mean_dists.append(float(torch.linalg.vector_norm(mean.detach() - target_mean.detach()).item()))
        cov_dists.append(float(torch.linalg.vector_norm((cov.detach() - target_cov.detach()).reshape(-1)).item()))
        active_classes += 1
    if not losses:
        return zero, empty
    loss = torch.stack(losses).mean()
    return loss, {
        "active_classes": int(active_classes),
        "samples": int(y.numel()),
        "mean_dist": float(sum(mean_dists) / max(1, len(mean_dists))),
        "cov_dist": float(sum(cov_dists) / max(1, len(cov_dists))),
        "skip_rate": float(1.0 - active_classes / max(1, total_classes)),
    }


class FedCVSVMBPrototypeBank:
    """Server-side TX/RX prototype bank for FedCVS-RFFI-VMB."""

    def __init__(self, num_classes: int, ema_alpha: float = 0.95, clip_norm: float = 1.0):
        self.num_classes = max(1, int(num_classes))
        self.ema_alpha = max(0.0, min(0.9999, float(ema_alpha)))
        self.clip_norm = float(clip_norm)
        self.tx_proto: Optional[torch.Tensor] = None
        self.tx_count = torch.zeros(self.num_classes, dtype=torch.float32)
        self.rx_proto: Dict[str, torch.Tensor] = {}
        self.rx_count: Dict[str, float] = {}
        self.rounds_updated = 0

    def _ensure_tx_shape(self, tx_sum: torch.Tensor) -> None:
        if self.tx_proto is None:
            self.tx_proto = torch.zeros(self.num_classes, int(tx_sum.size(1)), dtype=torch.float32)
            return
        if int(self.tx_proto.size(1)) != int(tx_sum.size(1)):
            raise ValueError(
                f"TX prototype dim mismatch: bank={int(self.tx_proto.size(1))} incoming={int(tx_sum.size(1))}"
            )

    def update(self, stats: Optional[FedCVSVMBPrototypeStats]) -> Dict[str, Any]:
        if stats is None:
            return self.summary()
        tx_sum = stats.tx_sum.detach().cpu().float()
        tx_count = stats.tx_count.detach().cpu().float()
        if tx_sum.dim() != 2:
            raise ValueError("tx_sum must be [num_classes, feature_dim]")
        if int(tx_sum.size(0)) != self.num_classes:
            raise ValueError(f"tx_sum num_classes mismatch: expected {self.num_classes}, got {int(tx_sum.size(0))}")
        self._ensure_tx_shape(tx_sum)
        assert self.tx_proto is not None

        for cls in range(self.num_classes):
            count = _as_client_count(tx_count[cls].item())
            if count <= 0.0:
                continue
            incoming = _clip_and_normalize(tx_sum[cls] / count, self.clip_norm)
            if float(self.tx_count[cls].item()) > 0.0:
                incoming = _safe_l2_normalize(
                    self.tx_proto[cls].float() * self.ema_alpha + incoming.float() * (1.0 - self.ema_alpha),
                    dim=0,
                )
            self.tx_proto[cls] = incoming.cpu()
            self.tx_count[cls] += float(count)

        for client_id, rx_sum in (stats.rx_sum_by_client or {}).items():
            count = _as_client_count((stats.rx_count_by_client or {}).get(client_id, 0.0))
            if count <= 0.0:
                continue
            incoming = _clip_and_normalize(rx_sum.detach().cpu().float() / count, self.clip_norm)
            cid = str(client_id)
            if cid in self.rx_proto:
                incoming = _safe_l2_normalize(
                    self.rx_proto[cid].float() * self.ema_alpha + incoming.float() * (1.0 - self.ema_alpha),
                    dim=0,
                )
            self.rx_proto[cid] = incoming.cpu()
            self.rx_count[cid] = float(self.rx_count.get(cid, 0.0)) + float(count)

        self.rounds_updated += 1
        return self.summary()

    def tx_tensors(self) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.tx_proto is None:
            return None, None
        return self.tx_proto.detach().clone(), self.tx_count.detach().clone()

    def rx_tensors(self, client_ids: Optional[Sequence[str]] = None) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], list[str]]:
        keys = [str(cid) for cid in (client_ids or sorted(self.rx_proto.keys())) if str(cid) in self.rx_proto]
        if not keys:
            return None, None, []
        proto = torch.stack([self.rx_proto[cid] for cid in keys], dim=0)
        count = torch.tensor([float(self.rx_count.get(cid, 0.0)) for cid in keys], dtype=torch.float32)
        return proto, count, keys

    def summary(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "rounds_updated": int(self.rounds_updated),
            "tx_count_nonzero": int((self.tx_count > 0).sum().item()),
            "rx_count_nonzero": int(len(self.rx_proto)),
            "tx_total_count": float(self.tx_count.sum().item()),
            "rx_total_count": float(sum(self.rx_count.values())),
        }


def merge_prototype_stats(stats_list: Iterable[Optional[FedCVSVMBPrototypeStats]]) -> Optional[FedCVSVMBPrototypeStats]:
    tx_sum = None
    tx_count = None
    rx_sum: Dict[str, torch.Tensor] = {}
    rx_count: Dict[str, float] = {}
    for stats in stats_list:
        if stats is None:
            continue
        tx_sum = stats.tx_sum.detach().cpu().float().clone() if tx_sum is None else tx_sum + stats.tx_sum.detach().cpu().float()
        tx_count = (
            stats.tx_count.detach().cpu().float().clone()
            if tx_count is None
            else tx_count + stats.tx_count.detach().cpu().float()
        )
        for cid, value in (stats.rx_sum_by_client or {}).items():
            key = str(cid)
            value_cpu = value.detach().cpu().float()
            rx_sum[key] = value_cpu.clone() if key not in rx_sum else rx_sum[key] + value_cpu
        for cid, value in (stats.rx_count_by_client or {}).items():
            key = str(cid)
            rx_count[key] = float(rx_count.get(key, 0.0)) + _as_client_count(value)
    if tx_sum is None or tx_count is None:
        return None
    return FedCVSVMBPrototypeStats(tx_sum=tx_sum, tx_count=tx_count, rx_sum_by_client=rx_sum, rx_count_by_client=rx_count)


def build_prototype_stats(
    z_t: Optional[torch.Tensor],
    z_r: Optional[torch.Tensor],
    labels: torch.Tensor,
    *,
    client_id: str,
    num_classes: int,
) -> Optional[FedCVSVMBPrototypeStats]:
    if not torch.is_tensor(z_t) or not torch.is_tensor(z_r):
        return None
    if z_t.dim() != 2 or z_r.dim() != 2 or int(z_t.size(0)) != int(labels.numel()) or int(z_r.size(0)) != int(labels.numel()):
        return None
    z_t_cpu = _safe_l2_normalize(z_t.detach().cpu(), dim=1)
    z_r_cpu = _safe_l2_normalize(z_r.detach().cpu(), dim=1)
    y = labels.detach().view(-1).cpu().long()
    c = max(1, int(num_classes))
    tx_sum = torch.zeros(c, int(z_t_cpu.size(1)), dtype=torch.float32)
    tx_count = torch.zeros(c, dtype=torch.float32)
    valid = (y >= 0) & (y < c)
    for cls in torch.unique(y[valid]):
        mask = valid & (y == cls)
        tx_sum[int(cls.item())] = z_t_cpu[mask].sum(dim=0)
        tx_count[int(cls.item())] = float(mask.sum().item())
    return FedCVSVMBPrototypeStats(
        tx_sum=tx_sum,
        tx_count=tx_count,
        rx_sum_by_client={str(client_id): z_r_cpu.sum(dim=0)},
        rx_count_by_client={str(client_id): float(z_r_cpu.size(0))},
    )


def prototype_contrastive_loss(
    features: Optional[torch.Tensor],
    labels: torch.Tensor,
    prototypes: Optional[torch.Tensor],
    counts: Optional[torch.Tensor],
    *,
    temperature: float = 0.1,
    min_count: int = 1,
) -> tuple[torch.Tensor, Dict[str, float]]:
    ref = features if torch.is_tensor(features) else None
    if ref is None:
        ref = prototypes if torch.is_tensor(prototypes) else torch.tensor(0.0)
    zero = ref.new_tensor(0.0)
    empty = {"active_prototypes": 0, "samples": 0, "target_cos": float("nan")}
    if not torch.is_tensor(features) or not torch.is_tensor(prototypes) or not torch.is_tensor(counts):
        return zero, empty
    if features.dim() != 2 or prototypes.dim() != 2 or int(prototypes.size(0)) == 0:
        return zero, empty
    counts = counts.to(device=features.device).view(-1).float()
    active = counts >= max(1, int(min_count))
    active = active & (torch.linalg.vector_norm(prototypes.to(device=features.device).float(), dim=1) > 0.0)
    active_indices = torch.nonzero(active, as_tuple=False).view(-1).long()
    if int(active_indices.numel()) < 2:
        return zero, empty
    labels = labels.to(device=features.device).view(-1).long()
    valid = torch.zeros_like(labels, dtype=torch.bool)
    target_pos = torch.full_like(labels, -1)
    for pos, proto_idx in enumerate(active_indices.tolist()):
        mask = labels == int(proto_idx)
        valid = valid | mask
        target_pos[mask] = int(pos)
    if not bool(valid.any()):
        return zero, {"active_prototypes": int(active_indices.numel()), "samples": 0, "target_cos": float("nan")}
    z = _safe_l2_normalize(features[valid], dim=1)
    proto = _safe_l2_normalize(prototypes.to(device=features.device, dtype=features.dtype)[active_indices], dim=1)
    temp = max(1e-6, float(temperature))
    logits = (z @ proto.t()) / temp
    target = target_pos[valid]
    loss = F.cross_entropy(logits.float(), target.long())
    cos = (z * proto[target]).sum(dim=1).mean()
    return loss, {
        "active_prototypes": int(active_indices.numel()),
        "samples": int(target.numel()),
        "target_cos": float(cos.detach().item()),
    }


def aggregate_gradients(
    client_gradients: Mapping[str, Mapping[str, torch.Tensor]],
    weights: Mapping[str, float],
) -> OrderedDict[str, torch.Tensor]:
    if not client_gradients:
        raise ValueError("client_gradients must not be empty")
    first = next(iter(client_gradients.values()))
    aggregated: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, ref in first.items():
        if not torch.is_tensor(ref):
            continue
        acc = torch.zeros_like(ref.detach().cpu())
        for cid, grads in client_gradients.items():
            if key not in grads:
                raise KeyError(f"Client {cid} is missing gradient key {key}")
            value = grads[key].detach().cpu()
            if tuple(value.shape) != tuple(ref.shape):
                raise ValueError(f"Gradient shape mismatch for {key}: {tuple(value.shape)} vs {tuple(ref.shape)}")
            acc.add_(value.to(dtype=acc.dtype), alpha=float(weights.get(cid, 0.0)))
        aggregated[key] = acc
    return aggregated


def gradient_norm(gradients: Mapping[str, torch.Tensor]) -> float:
    total = 0.0
    for grad in gradients.values():
        if torch.is_tensor(grad):
            total += float(torch.sum(grad.detach().float() * grad.detach().float()).item())
    return math.sqrt(max(0.0, total))


def gradient_payload_size_bytes(gradients: Mapping[str, torch.Tensor]) -> int:
    total = 0
    for grad in gradients.values():
        if torch.is_tensor(grad):
            total += int(grad.numel()) * int(grad.element_size())
    return int(total)


def prototype_stats_payload_size_bytes(stats: Optional[FedCVSVMBPrototypeStats]) -> int:
    if stats is None:
        return 0
    total = 0
    if torch.is_tensor(stats.tx_sum):
        total += int(stats.tx_sum.numel()) * int(stats.tx_sum.element_size())
    if torch.is_tensor(stats.tx_count):
        total += int(stats.tx_count.numel()) * int(stats.tx_count.element_size())
    for value in (stats.rx_sum_by_client or {}).values():
        if torch.is_tensor(value):
            total += int(value.numel()) * int(value.element_size())
    total += 8 * len(stats.rx_count_by_client or {})
    return int(total)


def coral_stats_payload_size_bytes(stats: Optional[FedCVSCoralStats]) -> int:
    if stats is None:
        return 0
    total = 0
    for value in (stats.class_sum, stats.class_outer, stats.class_count):
        if torch.is_tensor(value):
            total += int(value.numel()) * int(value.element_size())
    return int(total)


def gradient_cosine_summary(client_gradients: Mapping[str, Mapping[str, torch.Tensor]]) -> Dict[str, float]:
    ids = list(client_gradients.keys())
    if len(ids) < 2:
        return {"pairs": 0, "mean": float("nan"), "min": float("nan"), "max": float("nan")}
    values = []
    keys = sorted(set.intersection(*(set(client_gradients[cid].keys()) for cid in ids)))
    for i, cid_a in enumerate(ids):
        flat_a = torch.cat([client_gradients[cid_a][key].reshape(-1).float() for key in keys])
        for cid_b in ids[i + 1 :]:
            flat_b = torch.cat([client_gradients[cid_b][key].reshape(-1).float() for key in keys])
            denom = flat_a.norm().clamp_min(1e-12) * flat_b.norm().clamp_min(1e-12)
            values.append(float(torch.dot(flat_a, flat_b).div(denom).item()))
    return {
        "pairs": int(len(values)),
        "mean": float(sum(values) / max(1, len(values))),
        "min": float(min(values)) if values else float("nan"),
        "max": float(max(values)) if values else float("nan"),
    }


def apply_server_gradient_step(
    state: Mapping[str, torch.Tensor],
    gradients: Mapping[str, torch.Tensor],
    *,
    lr: float,
    momentum: float,
    weight_decay: float,
    optimizer_state: Optional[MutableMappingLike] = None,
) -> tuple[OrderedDict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, float]]:
    opt_state: Dict[str, torch.Tensor] = {
        str(k): v.detach().cpu().clone() for k, v in (optimizer_state or {}).items() if torch.is_tensor(v)
    }
    new_state: OrderedDict[str, torch.Tensor] = OrderedDict()
    updated = 0
    for key, value in state.items():
        cur = value.detach().cpu().clone()
        grad = gradients.get(key)
        if torch.is_tensor(grad) and torch.is_floating_point(cur):
            g = grad.detach().cpu().to(dtype=cur.dtype)
            if float(weight_decay) > 0.0:
                g = g + float(weight_decay) * cur
            if float(momentum) > 0.0:
                prev = opt_state.get(key, torch.zeros_like(g))
                buf = float(momentum) * prev.to(dtype=g.dtype) + g
            else:
                buf = g.clone()
            opt_state[key] = buf.detach().cpu().clone()
            cur = cur - float(lr) * buf.to(dtype=cur.dtype)
            updated += 1
        new_state[key] = cur
    return new_state, opt_state, {
        "updated_keys": float(updated),
        "grad_norm": gradient_norm(gradients),
        "server_lr": float(lr),
    }


MutableMappingLike = Mapping[str, torch.Tensor]


def vmb_stage_for_round(stage: str, round_idx: int, pretrain_rounds: int) -> str:
    mode = str(stage or "stage2").lower().strip()
    if mode == "auto":
        return "stage1" if int(round_idx) <= max(0, int(pretrain_rounds)) else "stage2"
    if mode not in {"stage1", "stage2"}:
        raise ValueError("fl_vmb_stage must be one of: auto, stage1, stage2")
    return mode


def is_stage2_rx_key(key: str) -> bool:
    return str(key).startswith(("dom_backbone.", "dom_head.", "dom_enhancer."))


def adversarial_warmup_weight(base_weight: float, *, round_idx: int, warmup_rounds: int, gamma: float = 10.0) -> float:
    base = float(base_weight)
    warm = int(warmup_rounds)
    if base <= 0.0 or warm <= 0 or int(round_idx) >= warm:
        return base
    p = max(0.0, min(1.0, float(round_idx) / float(max(1, warm))))
    return base * (2.0 / (1.0 + math.exp(-float(gamma) * p)) - 1.0)
