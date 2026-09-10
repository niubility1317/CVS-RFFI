"""Per-record readouts followed by strictly disjoint clean donor aggregation."""
import torch
from torch import nn


class ResponseReadouts(nn.Module):
    def __init__(self, id_dim, dom_dim, readout_dim=16):
        super().__init__()
        self.tx = nn.Linear(id_dim, readout_dim)
        self.rx = nn.Linear(dom_dim, readout_dim)

    def forward(self, id_grid, dom_grid, query_tx, query_rx, donor_tx, donor_rx,
                detach_identity=False):
        if id_grid.ndim != 4 or dom_grid.ndim != 4 or id_grid.shape[:3] != dom_grid.shape[:3]:
            raise ValueError('clean feature grids must be [P,Q,K,D]')
        p, q, k = id_grid.shape[:3]
        if k < 1:
            raise ValueError('empty physical-record axis')
        def indices(values, size):
            v = torch.as_tensor(values, device=id_grid.device, dtype=torch.long)
            if v.ndim != 1 or not v.numel() or v.unique().numel() != v.numel() or (v < 0).any() or (v >= size).any():
                raise ValueError('role indices must be unique, nonempty and in bounds')
            return v
        yt, yr, dt, dr = indices(query_tx,p), indices(query_rx,q), indices(donor_tx,p), indices(donor_rx,q)
        if torch.isin(yt,dt).any() or torch.isin(yr,dr).any():
            raise ValueError('query and donor roles overlap')
        zi = id_grid.index_select(0,yt).index_select(1,dr)
        zd = dom_grid.index_select(0,dt).index_select(1,yr)
        if detach_identity:
            zi = zi.detach()
        return self.tx(zi).mean((1,2)), self.rx(zd).mean((0,2))
