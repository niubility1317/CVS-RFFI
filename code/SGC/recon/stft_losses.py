from __future__ import annotations

import torch
import torch.nn.functional as F

from .complex_ops import iq_to_complex


def complex_stft(
    x: torch.Tensor,
    *,
    n_fft: int = 64,
    hop_length: int = 16,
    win_length: int = 64,
) -> torch.Tensor:
    z = iq_to_complex(x)
    window = torch.hann_window(int(win_length), device=x.device, dtype=x.float().dtype)
    return torch.stft(
        z,
        n_fft=int(n_fft),
        hop_length=int(hop_length),
        win_length=int(win_length),
        window=window,
        return_complex=True,
        center=True,
    )


def stft_l1_loss(
    x_hat: torch.Tensor,
    target: torch.Tensor,
    *,
    n_fft: int = 64,
    hop_length: int = 16,
    win_length: int = 64,
) -> torch.Tensor:
    Xh = complex_stft(x_hat, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    Xt = complex_stft(target, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    return F.l1_loss(Xh.real, Xt.real) + F.l1_loss(Xh.imag, Xt.imag)


def stft_mag_phase_loss(
    x_hat: torch.Tensor,
    target: torch.Tensor,
    *,
    n_fft: int = 64,
    hop_length: int = 16,
    win_length: int = 64,
    phase_weight: float = 0.1,
    eps: float = 1e-6,
) -> torch.Tensor:
    Xh = complex_stft(x_hat, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    Xt = complex_stft(target, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    loss_mag = F.l1_loss(torch.log(Xh.abs() + float(eps)), torch.log(Xt.abs() + float(eps)))
    phase_diff = torch.angle(Xh * torch.conj(Xt))
    loss_phase = (1.0 - torch.cos(phase_diff)).mean()
    return loss_mag + float(phase_weight) * loss_phase
