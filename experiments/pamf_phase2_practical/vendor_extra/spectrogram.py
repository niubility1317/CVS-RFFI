from __future__ import annotations

import torch


def ensure_iq_2xl(iq: torch.Tensor) -> torch.Tensor:
    """Normalize IQ input to `[B, 2, L]` float tensor."""

    if not torch.is_tensor(iq):
        iq = torch.as_tensor(iq)
    if torch.is_complex(iq):
        if iq.dim() == 1:
            iq = iq.unsqueeze(0)
        return torch.stack([iq.real, iq.imag], dim=1).float()
    iq = iq.float()
    if iq.dim() == 2 and iq.size(0) == 2:
        iq = iq.unsqueeze(0)
    if iq.dim() != 3 or iq.size(1) != 2:
        raise ValueError(f"Expected complex [L], [2,L], or [B,2,L], got {tuple(iq.shape)}")
    return iq


def iq_to_complex(iq: torch.Tensor) -> torch.Tensor:
    iq = ensure_iq_2xl(iq)
    return torch.complex(iq[:, 0], iq[:, 1])


def iq_to_log_spectrogram(
    iq: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int | None = None,
    window: str = "hann",
    eps: float = 1e-8,
    normalize: bool | str = True,
) -> torch.Tensor:
    """Convert IQ samples to log-amplitude spectrogram `[B,1,F,T]`."""

    x = iq_to_complex(iq)
    win_length = int(win_length or n_fft)
    if window == "hann":
        win = torch.hann_window(win_length, device=x.device, dtype=x.real.dtype)
    elif window in ("none", None):
        win = None
    else:
        raise ValueError(f"Unsupported window={window!r}")

    stft = torch.stft(
        x,
        n_fft=int(n_fft),
        hop_length=int(hop_length),
        win_length=win_length,
        window=win,
        center=True,
        return_complex=True,
        onesided=False,
    )
    spec = torch.log(torch.abs(stft).clamp_min(float(eps)))
    norm = "zscore" if normalize is True else ("none" if normalize is False else str(normalize))
    if norm == "zscore":
        mean = spec.mean(dim=(-2, -1), keepdim=True)
        std = spec.std(dim=(-2, -1), keepdim=True).clamp_min(float(eps))
        spec = (spec - mean) / std
    elif norm == "minmax":
        lo = spec.amin(dim=(-2, -1), keepdim=True)
        hi = spec.amax(dim=(-2, -1), keepdim=True)
        spec = (spec - lo) / (hi - lo).clamp_min(float(eps))
    elif norm != "none":
        raise ValueError(f"Unsupported spectrogram normalize={normalize!r}")
    return spec.unsqueeze(1).float()
