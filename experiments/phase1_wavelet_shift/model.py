"""Randomize the Haar grid only during source training."""
import torch
from experiments.phase1_wavelet_lifting.model import WaveletLifting

class ShiftAugmentedWavelet(WaveletLifting):
    def forward(self, received, initial, log_variance):
        if not self.training:
            return super().forward(received,initial,log_variance)
        shift=int(torch.randint(0,16,(),device=received.device).item())
        result=super().forward(torch.roll(received,shift,dims=2),
                               torch.roll(initial,shift,dims=2),log_variance)
        return torch.roll(result,-shift,dims=2)
