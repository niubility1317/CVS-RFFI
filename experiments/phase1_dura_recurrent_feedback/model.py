"""Three shared-weight continuous refinements of a DURA waveform."""
import torch
from torch import nn
from experiments.phase1_dura_feedback.model import FeedbackRecovery


class RecurrentFeedbackRecovery(nn.Module):
    def __init__(self):
        super().__init__()
        self.step = FeedbackRecovery()

    @property
    def entry(self):
        return self.step.entry

    @property
    def gate_output(self):
        return self.step.gate_output

    def forward(self, received, packed, log_variance):
        n = len(received)
        if received.shape != (n, 2, 256) or packed.shape != (n, 3, 256) or log_variance.shape != (n,):
            raise ValueError('Matching IQ, frozen condition and blind variance required')
        if not all(torch.isfinite(v).all() for v in (received, packed, log_variance)):
            raise FloatingPointError('Finite same-window input required')
        wave = packed[:, :2]
        condition = packed[:, 2:3].detach()
        for _ in range(3):
            wave = self.step(received, torch.cat((wave, condition), dim=1), log_variance)
        return wave
