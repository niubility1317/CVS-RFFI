from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import torch
import torch.nn.functional as F


def excitation_descriptors(iq: torch.Tensor) -> torch.Tensor:
    if iq.ndim != 3 or int(iq.shape[1]) != 2:
        raise ValueError("iq must have shape [batch,2,time]")
    x = torch.complex(iq[:, 0].float(), iq[:, 1].float())
    amplitude = x.abs().clamp_min(1e-8)
    power = amplitude.square()
    papr = power.max(dim=1).values / power.mean(dim=1).clamp_min(1e-8)
    q50 = torch.quantile(amplitude, 0.50, dim=1)
    q90 = torch.quantile(amplitude, 0.90, dim=1)
    spectrum = torch.fft.fft(x, dim=1).abs().square()
    occupancy = (spectrum > spectrum.mean(dim=1, keepdim=True)).float().mean(dim=1)
    slew = (x[:, 1:] - x[:, :-1]).abs().mean(dim=1)
    circularity = x.square().mean(dim=1).abs() / power.mean(dim=1).clamp_min(1e-8)
    return torch.stack((papr, q50, q90, occupancy + slew, circularity), dim=1)


def receiver_conditioned_alignment_loss(
    features: torch.Tensor,
    *,
    tx: torch.Tensor,
    receiver: torch.Tensor,
    excitation_bin: torch.Tensor,
) -> dict[str, Any]:
    """Align receivers only within the same TX and excitation bin."""

    features = F.normalize(features.float(), dim=-1)
    tx = tx.reshape(-1).long().to(features.device)
    receiver = receiver.reshape(-1).long().to(features.device)
    excitation_bin = excitation_bin.reshape(-1).long().to(features.device)
    if features.ndim != 2 or not (features.shape[0] == tx.numel() == receiver.numel() == excitation_bin.numel()):
        raise ValueError("receiver-conditioned alignment inputs must align")
    grouped: dict[tuple[int, int], list[torch.Tensor]] = defaultdict(list)
    for tx_id, rx_id, bin_id in set(zip(tx.tolist(), receiver.tolist(), excitation_bin.tolist())):
        selected = tx.eq(tx_id) & receiver.eq(rx_id) & excitation_bin.eq(bin_id)
        grouped[(int(tx_id), int(bin_id))].append(F.normalize(features[selected].mean(dim=0), dim=0))
    losses = []
    pair_count = 0
    for prototypes in grouped.values():
        if len(prototypes) < 2:
            continue
        for left in range(len(prototypes) - 1):
            for right in range(left + 1, len(prototypes)):
                losses.append(1.0 - (prototypes[left] * prototypes[right]).sum())
                pair_count += 1
    loss = torch.stack(losses).mean() if losses else features.sum() * 0.0
    return {"loss": loss, "pair_count": pair_count}


class ReceiverConditionedPrototypeBank:
    def __init__(self, *, feature_dim: int, min_count: int = 2, momentum: float = 0.9) -> None:
        if int(feature_dim) < 1 or int(min_count) < 1 or not 0.0 <= float(momentum) < 1.0:
            raise ValueError("invalid conditioned prototype configuration")
        self.feature_dim = int(feature_dim)
        self.min_count = int(min_count)
        self.momentum = float(momentum)
        self._entries: dict[tuple[int, int, int], tuple[torch.Tensor, int]] = {}

    def update(
        self,
        features: torch.Tensor,
        *,
        tx: torch.Tensor,
        receiver: torch.Tensor,
        excitation_bin: torch.Tensor,
    ) -> None:
        features = features.detach().float()
        tx = tx.reshape(-1).long()
        receiver = receiver.reshape(-1).long()
        excitation_bin = excitation_bin.reshape(-1).long()
        if features.ndim != 2 or features.shape[1] != self.feature_dim or not (
            features.shape[0] == tx.numel() == receiver.numel() == excitation_bin.numel()
        ):
            raise ValueError("conditioned prototype inputs must align")
        for tx_id, rx_id, bin_id in set(zip(tx.tolist(), receiver.tolist(), excitation_bin.tolist())):
            mask = tx.eq(tx_id) & receiver.eq(rx_id) & excitation_bin.eq(bin_id)
            value = F.normalize(features[mask].mean(dim=0), dim=0)
            key = (int(tx_id), int(rx_id), int(bin_id))
            old = self._entries.get(key)
            if old is None:
                self._entries[key] = (value.cpu(), int(mask.sum()))
            else:
                mixed = F.normalize(self.momentum * old[0] + (1.0 - self.momentum) * value.cpu(), dim=0)
                self._entries[key] = (mixed, old[1] + int(mask.sum()))

    def alignment_loss(self) -> dict[str, Any]:
        grouped: dict[tuple[int, int], list[torch.Tensor]] = defaultdict(list)
        for (tx_id, _rx_id, bin_id), (prototype, count) in self._entries.items():
            if count >= self.min_count:
                grouped[(tx_id, bin_id)].append(prototype)
        losses = []
        pair_count = 0
        for prototypes in grouped.values():
            if len(prototypes) < 2:
                continue
            stacked = torch.stack(prototypes)
            center = F.normalize(stacked.mean(dim=0), dim=0)
            losses.append((1.0 - (stacked * center).sum(dim=1)).mean())
            pair_count += 1
        loss = torch.stack(losses).mean() if losses else torch.tensor(0.0)
        return {"loss": loss, "pair_count": pair_count}


class GroupTailRiskEMA:
    def __init__(
        self,
        *,
        alpha: float,
        momentum: float,
        min_group_size: int,
        max_weight: float,
    ) -> None:
        if not 0.0 < float(alpha) <= 1.0 or not 0.0 <= float(momentum) < 1.0:
            raise ValueError("invalid group tail risk schedule")
        if int(min_group_size) < 1 or float(max_weight) < 1.0:
            raise ValueError("invalid group eligibility or weight cap")
        self.alpha = float(alpha)
        self.momentum = float(momentum)
        self.min_group_size = int(min_group_size)
        self.max_weight = float(max_weight)
        self._ema: dict[int, float] = {}

    def update(self, *, group_ids: torch.Tensor, losses: torch.Tensor) -> dict[str, Any]:
        group_ids = group_ids.reshape(-1).long()
        losses = losses.reshape(-1).float()
        if group_ids.numel() != losses.numel():
            raise ValueError("group ids and losses must align")
        eligible = []
        for group_id in group_ids.unique(sorted=True).tolist():
            selected = group_ids.eq(group_id)
            if int(selected.sum()) < self.min_group_size:
                continue
            current = float(losses[selected].mean().detach().item())
            old = self._ema.get(int(group_id), current)
            value = self.momentum * old + (1.0 - self.momentum) * current
            self._ema[int(group_id)] = value
            eligible.append((int(group_id), value))
        if not eligible:
            return {
                "loss": losses.sum() * 0.0,
                "eligible_group_count": 0,
                "selected_group_ids": torch.empty(0, dtype=torch.long, device=losses.device),
                "max_effective_weight": torch.tensor(0.0, device=losses.device),
            }
        count = max(1, int(math.ceil(self.alpha * len(eligible))))
        selected_groups = sorted(eligible, key=lambda item: item[1], reverse=True)[:count]
        selected_ids = torch.tensor([item[0] for item in selected_groups], device=losses.device)
        mask = torch.zeros_like(losses, dtype=torch.bool)
        for group_id in selected_ids.tolist():
            mask |= group_ids.eq(group_id)
        per_group_weight = min(self.max_weight, float(len(eligible)) / float(count))
        return {
            "loss": losses[mask].mean(),
            "eligible_group_count": len(eligible),
            "selected_group_ids": selected_ids,
            "max_effective_weight": torch.tensor(per_group_weight, device=losses.device),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "momentum": self.momentum,
            "min_group_size": self.min_group_size,
            "max_weight": self.max_weight,
            "ema": dict(self._ema),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.alpha = float(state["alpha"])
        self.momentum = float(state["momentum"])
        self.min_group_size = int(state["min_group_size"])
        self.max_weight = float(state["max_weight"])
        self._ema = {int(key): float(value) for key, value in dict(state.get("ema", {})).items()}
