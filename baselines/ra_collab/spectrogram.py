from __future__ import annotations

import torch

from baselines.common.spectrogram import iq_to_log_spectrogram


class SpectrogramTransform:
    """Local spectrogram transform for the RA-Collab RFFI baseline."""

    def __init__(self, n_fft: int = 128, hop_length: int = 64, win_length: int | None = None, normalize: str = "zscore"):
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length or n_fft)
        self.normalize = normalize

    def __call__(self, iq: torch.Tensor) -> torch.Tensor:
        was_single = torch.is_tensor(iq) and iq.dim() == 2
        spec = iq_to_log_spectrogram(
            iq,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            normalize=self.normalize,
        )
        return spec[0] if was_single else spec
