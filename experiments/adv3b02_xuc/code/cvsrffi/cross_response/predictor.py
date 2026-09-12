"""Shared, capacity-limited response prediction without equipment lookups."""
import torch
from torch import nn


class SharedResponsePredictor(nn.Module):
    def __init__(self, readout_dim, target_dim, mode='additive', rank=4):
        super().__init__()
        if mode not in ('additive', 'bilinear') or min(readout_dim, target_dim, rank) < 1:
            raise ValueError('invalid predictor capacity or mode')
        self.mode = mode
        self.tx = nn.Linear(readout_dim, target_dim, bias=False)
        self.rx = nn.Linear(readout_dim, target_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(target_dim))
        if mode == 'bilinear':
            self.bt = nn.Linear(readout_dim, rank, bias=False)
            self.br = nn.Linear(readout_dim, rank, bias=False)
            self.u = nn.Linear(rank, target_dim, bias=False)

    def forward(self, tx, rx):
        if tx.ndim != 2 or rx.ndim != 2 or min(tx.shape[0], rx.shape[0]) < 1:
            raise ValueError('expected nonempty descriptor matrices')
        out = self.bias + self.tx(tx)[:, None] + self.rx(rx)[None, :]
        if self.mode == 'bilinear':
            out = out + self.u(self.bt(tx)[:, None] * self.br(rx)[None, :])
        return out

    def centered_interaction(self, tx, rx):
        if self.mode == 'additive':
            return self(tx, rx) * 0
        a, b = self.bt(tx), self.br(rx)
        return self.u((a-a.mean(0))[:, None]*(b-b.mean(0))[None, :])
