from __future__ import annotations

import os
from typing import Dict, List

import torch


class AveragedModelState:
    """EMA/SWA/SWAD-style online weight averaging."""

    def __init__(self, mode: str, decay: float = 0.999):
        self.mode = str(mode)
        self.decay = float(decay)
        self.n = 0
        self.avg: Dict[str, torch.Tensor] = {}
        self.non_float: Dict[str, torch.Tensor] = {}
        self.epochs: List[int] = []

    def update(self, model, epoch: int, *, ema: bool = False) -> None:
        state = getattr(model, "_orig_mod", model).state_dict()
        with torch.no_grad():
            for k, v in state.items():
                vv = v.detach()
                if torch.is_floating_point(vv):
                    vf = vv.float().clone()
                    if k not in self.avg:
                        self.avg[k] = vf
                    elif ema:
                        self.avg[k].mul_(float(self.decay)).add_(vf, alpha=1.0 - float(self.decay))
                    else:
                        self.avg[k].mul_(float(self.n) / float(self.n + 1)).add_(vf, alpha=1.0 / float(self.n + 1))
                else:
                    self.non_float[k] = vv.clone()
        self.n += 1
        self.epochs.append(int(epoch))

    def has_state(self) -> bool:
        return self.n > 0 and len(self.avg) > 0

    def averaged_state_dict(self, model) -> Dict[str, torch.Tensor]:
        ref_state = getattr(model, "_orig_mod", model).state_dict()
        out = {}
        for k, v in ref_state.items():
            if k in self.avg:
                out[k] = self.avg[k].to(device=v.device, dtype=v.dtype)
            elif k in self.non_float:
                out[k] = self.non_float[k].to(device=v.device, dtype=v.dtype)
            else:
                out[k] = v.detach().clone()
        return out

    def cpu_state_dict(self, model) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.averaged_state_dict(model).items()}


def save_checkpoint(path: str, *, model, optimizer, scheduler, scaler, epoch: int, args, split_info, stats: dict):
    parent = os.path.dirname(os.path.abspath(str(path)))
    if parent:
        os.makedirs(parent, exist_ok=True)
    state_model = getattr(model, "_orig_mod", model)
    payload = {
        "model": state_model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "args": vars(args),
        "split_info": split_info,
        "stats": stats,
    }
    torch.save(payload, path)


def derive_checkpoint_path(base_path: str, suffix: str) -> str:
    """Derive a checkpoint path when user does not provide one explicitly.

    Example:
      best_model.pth + test_overall -> best_model_test_overall.pth
    """
    base_path = str(base_path).strip() or "best_model.pth"
    root, ext = os.path.splitext(base_path)
    if ext == "":
        ext = ".pth"
    return f"{root}_{suffix}{ext}"


def default_is_path(p: str, default_name: str) -> bool:
    return str(p).strip() == default_name
