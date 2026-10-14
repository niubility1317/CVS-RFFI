import itertools
import torch
from . import design as d
from experiments.cvs_backbone_fishr50.objective import validate_layout


def angle(G, y, rx, day, ids, role='L_s'):
    validate_layout(y, rx, day, ids, role)
    if G.ndim != 2 or len(G) != 180 or not G.shape[1] or not torch.isfinite(G).all():
        raise ValueError('Finite actual source parameter-gradient matrix required')
    G = G.double()
    v = torch.stack([G[rx == receiver].var(0, unbiased=True) for receiver in d.parent.RX])
    norms = v.norm(dim=1)
    if not torch.isfinite(norms).all() or (norms <= d.EPS).any():
        raise ValueError('Variance-vector direction is not identifiable')
    unit = v / norms[:, None]
    pairs = torch.stack([1 - (unit[i] * unit[j]).sum()
                         for i, j in itertools.combinations(range(5), 2)])
    loss = pairs.mean()
    if not torch.isfinite(loss) or loss < -1e-12 or loss > 1 + 1e-12:
        raise ValueError('Invalid nonnegative variance-vector angle')
    return loss


def stats(a, b):
    a, b = a.double(), b.double()
    na, nb = a.norm(), b.norm()
    return dict(CE_norm=float(na), auxiliary_norm=float(nb),
                auxiliary_over_CE=float(nb / na) if na > 0 else None,
                cosine=float((a @ b) / (na * nb)) if na > 0 and nb > 0 else None)
