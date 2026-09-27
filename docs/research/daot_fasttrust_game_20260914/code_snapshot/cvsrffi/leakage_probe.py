from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F


def frozen_ridge_linear_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    eval_features: torch.Tensor,
    eval_labels: torch.Tensor,
    *,
    ridge: float = 1e-2,
) -> Dict[str, Any]:
    """Fit a deterministic frozen linear probe and evaluate on a disjoint split."""

    default = {
        "status": "FAILED",
        "reason": "invalid_input",
        "train_count": 0,
        "eval_count": 0,
        "class_count": 0,
        "accuracy": float("nan"),
        "balanced_accuracy": float("nan"),
        "chance_accuracy": float("nan"),
        "excess_accuracy": float("nan"),
        "balanced_chance_accuracy": float("nan"),
        "balanced_excess_accuracy": float("nan"),
        "ridge": float(ridge),
    }
    if any(value is None or not torch.is_tensor(value) for value in (train_features, train_labels, eval_features, eval_labels)):
        return default
    if train_features.dim() != 2 or eval_features.dim() != 2 or train_features.size(1) != eval_features.size(1):
        return default
    train_y = train_labels.detach().view(-1).long().cpu()
    eval_y = eval_labels.detach().view(-1).long().cpu()
    train_x = train_features.detach().float().cpu()
    eval_x = eval_features.detach().float().cpu()
    if train_x.size(0) != train_y.numel() or eval_x.size(0) != eval_y.numel():
        return default
    train_valid = train_y >= 0
    eval_valid = eval_y >= 0
    train_x, train_y = train_x[train_valid], train_y[train_valid]
    eval_x, eval_y = eval_x[eval_valid], eval_y[eval_valid]
    classes = torch.unique(train_y, sorted=True)
    if train_x.size(0) < 2 or eval_x.size(0) < 1 or classes.numel() < 2:
        default.update(
            {
                "reason": "insufficient_samples_or_classes",
                "train_count": int(train_x.size(0)),
                "eval_count": int(eval_x.size(0)),
                "class_count": int(classes.numel()),
            }
        )
        return default
    class_to_index = {int(cls): idx for idx, cls in enumerate(classes.tolist())}
    eval_known = torch.tensor([int(v) in class_to_index for v in eval_y.tolist()], dtype=torch.bool)
    eval_x, eval_y = eval_x[eval_known], eval_y[eval_known]
    if eval_x.size(0) < 1:
        default.update(
            {
                "reason": "no_eval_labels_seen_in_train",
                "train_count": int(train_x.size(0)),
                "class_count": int(classes.numel()),
            }
        )
        return default

    mean = train_x.mean(dim=0, keepdim=True)
    scale = train_x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-5)
    train_x = F.normalize((train_x - mean) / scale, dim=1)
    eval_x = F.normalize((eval_x - mean) / scale, dim=1)
    train_x = torch.cat([train_x, torch.ones((train_x.size(0), 1), dtype=train_x.dtype)], dim=1).double()
    eval_x = torch.cat([eval_x, torch.ones((eval_x.size(0), 1), dtype=eval_x.dtype)], dim=1).double()
    train_index = torch.tensor([class_to_index[int(v)] for v in train_y.tolist()], dtype=torch.long)
    target = F.one_hot(train_index, num_classes=int(classes.numel())).double()
    gram = train_x.t().matmul(train_x)
    penalty = torch.eye(gram.size(0), dtype=gram.dtype) * max(1e-8, float(ridge))
    penalty[-1, -1] = 0.0
    try:
        weight = torch.linalg.solve(gram + penalty, train_x.t().matmul(target))
    except RuntimeError:
        weight = torch.linalg.pinv(gram + penalty).matmul(train_x.t()).matmul(target)
    pred_index = eval_x.matmul(weight).argmax(dim=1)
    pred = classes[pred_index]
    accuracy = float(pred.eq(eval_y).float().mean().item())
    class_acc = []
    for cls in torch.unique(eval_y, sorted=True):
        mask = eval_y.eq(cls)
        if bool(mask.any()):
            class_acc.append(pred[mask].eq(eval_y[mask]).float().mean())
    balanced = float(torch.stack(class_acc).mean().item()) if class_acc else float("nan")
    counts = torch.stack([eval_y.eq(cls).sum() for cls in torch.unique(eval_y, sorted=True)]).float()
    chance = float((counts.max() / counts.sum().clamp_min(1.0)).item())
    balanced_chance = 1.0 / float(max(1, len(class_acc)))
    return {
        "status": "COMPLETE",
        "reason": "",
        "train_count": int(train_x.size(0)),
        "eval_count": int(eval_x.size(0)),
        "class_count": int(classes.numel()),
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "chance_accuracy": chance,
        "excess_accuracy": accuracy - chance,
        "balanced_chance_accuracy": balanced_chance,
        "balanced_excess_accuracy": balanced - balanced_chance,
        "ridge": float(ridge),
    }
