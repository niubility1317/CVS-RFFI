from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional

import torch
import torch.nn.functional as F


@dataclass
class LogitAnchorStats:
    logit_sum: torch.Tensor
    count: torch.Tensor
    teacher_correct: float
    teacher_total: float


def build_logit_anchor_stats(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    confidence_min: float = 0.0,
    margin_min: float = 0.0,
    require_correct: bool = True,
) -> Optional[LogitAnchorStats]:
    if not torch.is_tensor(logits) or logits.dim() != 2 or logits.numel() == 0:
        return None
    y = labels.detach().view(-1).long().to(device=logits.device)
    if int(y.numel()) != int(logits.size(0)):
        return None
    prob = F.softmax(logits.detach().float(), dim=1)
    topk = torch.topk(prob, k=min(2, int(prob.size(1))), dim=1)
    pred = topk.indices[:, 0]
    conf = topk.values[:, 0]
    margin = topk.values[:, 0] - (topk.values[:, 1] if topk.values.size(1) > 1 else 0.0)
    valid = (y >= 0) & (y < int(logits.size(1)))
    correct = pred == y
    if bool(require_correct):
        valid = valid & correct
    valid = valid & (conf >= float(confidence_min)) & (margin >= float(margin_min))
    if not bool(valid.any()):
        return None
    c = int(logits.size(1))
    logit_sum = torch.zeros(c, c, dtype=torch.float32, device=logits.device)
    count = torch.zeros(c, dtype=torch.float32, device=logits.device)
    for cls in torch.unique(y[valid]):
        mask = valid & (y == cls)
        logit_sum[int(cls.item())] = logits.detach().float()[mask].sum(dim=0)
        count[int(cls.item())] = float(mask.sum().item())
    return LogitAnchorStats(
        logit_sum=logit_sum.cpu(),
        count=count.cpu(),
        teacher_correct=float(correct[valid].sum().detach().cpu().item()),
        teacher_total=float(valid.sum().detach().cpu().item()),
    )


def merge_logit_anchor_stats(stats_list: Iterable[Optional[LogitAnchorStats]]) -> Optional[LogitAnchorStats]:
    logit_sum = None
    count = None
    teacher_correct = 0.0
    teacher_total = 0.0
    for stats in stats_list:
        if stats is None:
            continue
        logit_sum = stats.logit_sum.detach().cpu().float().clone() if logit_sum is None else logit_sum + stats.logit_sum.detach().cpu().float()
        count = stats.count.detach().cpu().float().clone() if count is None else count + stats.count.detach().cpu().float()
        teacher_correct += float(stats.teacher_correct)
        teacher_total += float(stats.teacher_total)
    if logit_sum is None or count is None:
        return None
    return LogitAnchorStats(logit_sum=logit_sum, count=count, teacher_correct=teacher_correct, teacher_total=teacher_total)


class LogitAnchorBank:
    def __init__(self, num_classes: int, ema_alpha: float = 0.9):
        self.num_classes = max(1, int(num_classes))
        self.ema_alpha = max(0.0, min(0.9999, float(ema_alpha)))
        self.anchor_logits = torch.zeros(self.num_classes, self.num_classes, dtype=torch.float32)
        self.count = torch.zeros(self.num_classes, dtype=torch.float32)
        self.rounds_updated = 0
        self.teacher_correct = 0.0
        self.teacher_total = 0.0

    def update(self, stats: Optional[LogitAnchorStats]) -> Dict[str, Any]:
        if stats is None:
            return self.summary()
        logit_sum = stats.logit_sum.detach().cpu().float()
        count = stats.count.detach().cpu().float()
        if tuple(logit_sum.shape) != tuple(self.anchor_logits.shape):
            raise ValueError(f"logit anchor shape mismatch: {tuple(logit_sum.shape)} vs {tuple(self.anchor_logits.shape)}")
        for cls in range(self.num_classes):
            n = float(count[cls].item())
            if n <= 0.0:
                continue
            incoming = logit_sum[cls] / max(1.0, n)
            if float(self.count[cls].item()) > 0.0:
                incoming = self.anchor_logits[cls] * self.ema_alpha + incoming * (1.0 - self.ema_alpha)
            self.anchor_logits[cls] = incoming
            self.count[cls] += n
        self.teacher_correct += float(stats.teacher_correct)
        self.teacher_total += float(stats.teacher_total)
        self.rounds_updated += 1
        return self.summary()

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.anchor_logits.detach().clone(), self.count.detach().clone()

    def summary(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "rounds_updated": int(self.rounds_updated),
            "anchor_count_nonzero": int((self.count > 0).sum().item()),
            "anchor_total_count": float(self.count.sum().item()),
            "teacher_correct_rate": float(self.teacher_correct / max(1.0, self.teacher_total)),
            "payload_bytes": int(self.anchor_logits.numel() * self.anchor_logits.element_size() + self.count.numel() * self.count.element_size()),
        }


def logit_anchor_kd_loss(
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    anchor_logits: Optional[torch.Tensor],
    counts: Optional[torch.Tensor],
    *,
    temperature: float = 2.0,
    min_count: int = 1,
) -> tuple[torch.Tensor, Dict[str, float]]:
    zero = student_logits.new_tensor(0.0)
    empty = {"kd_active": 0.0, "anchor_count": 0.0, "teacher_correct_rate": float("nan")}
    if not torch.is_tensor(anchor_logits) or not torch.is_tensor(counts):
        return zero, empty
    y = labels.view(-1).long().to(device=student_logits.device)
    if int(y.numel()) != int(student_logits.size(0)):
        return zero, empty
    counts = counts.to(device=student_logits.device).float().view(-1)
    valid = (y >= 0) & (y < int(anchor_logits.size(0)))
    if bool(valid.any()):
        valid = valid & (counts[y.clamp(0, int(anchor_logits.size(0)) - 1)] >= max(1, int(min_count)))
    if not bool(valid.any()):
        return zero, empty
    temp = max(1e-6, float(temperature))
    teacher = anchor_logits.to(device=student_logits.device, dtype=student_logits.dtype)[y[valid]]
    teacher_prob = F.softmax(teacher.float() / temp, dim=1)
    student_log_prob = F.log_softmax(student_logits[valid].float() / temp, dim=1)
    loss = F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * (temp * temp)
    return loss, {
        "kd_active": 1.0,
        "anchor_count": float((counts > 0).sum().detach().cpu().item()),
        "teacher_correct_rate": float("nan"),
    }


def logit_anchor_stats_payload_size_bytes(stats: Optional[LogitAnchorStats]) -> int:
    if stats is None:
        return 0
    total = 0
    total += int(stats.logit_sum.numel()) * int(stats.logit_sum.element_size())
    total += int(stats.count.numel()) * int(stats.count.element_size())
    total += 16
    return int(total)
