from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F

from .complex_ops import residual_safety_loss
from .stft_losses import stft_l1_loss


def channel_consistency_loss(
    channel,
    x_hat: torch.Tensor,
    y_sat: torch.Tensor,
    phi: Mapping[str, object],
    *,
    tf_weight: float = 0.3,
    n_fft: int = 64,
    hop_length: int = 16,
    win_length: int = 64,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    y_reproj = channel(x_hat, phi)
    loss_time = F.l1_loss(y_reproj, y_sat)
    loss_tf = stft_l1_loss(y_reproj, y_sat, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    loss = loss_time + float(tf_weight) * loss_tf
    return loss, {
        "loss_chan_time": loss_time.detach(),
        "loss_chan_tf": loss_tf.detach(),
        "loss_chan": loss.detach(),
    }


def residual_constraint_loss(x_hat: torch.Tensor, y: torch.Tensor, r_max: float = 0.15) -> torch.Tensor:
    return residual_safety_loss(x_hat, y, r_max=r_max)
