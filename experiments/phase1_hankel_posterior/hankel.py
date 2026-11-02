"""Per-window low-rank Hankel shrinkage of a physical-posterior IQ estimate."""

import math
import torch


@torch.no_grad()
def hankel_shrink(initial, observation_variance, *, window=32, shrink_scale=0.1, mix=0.5):
    """Apply fixed conservative shrinkage without identity labels or external templates.

    The blind observation variance determines scale, not a claim of calibrated
    posterior uncertainty. Hankel anti-diagonals are averaged exactly.
    """
    if (initial.ndim != 3 or initial.shape[1:] != (2, 256)
            or observation_variance.shape != (len(initial),) or not 1 <= window <= 256
            or not 0 <= mix <= 1 or shrink_scale < 0
            or not torch.isfinite(initial).all() or not torch.isfinite(observation_variance).all()
            or (observation_variance < 0).any()):
        raise ValueError('Finite IQ windows, nonnegative variances and valid fixed parameters required')
    if mix == 0 or shrink_scale == 0 or not (observation_variance > 0).any():
        return initial.clone()
    z = torch.complex(initial[:, 0], initial[:, 1])
    columns = 256-window+1
    matrix = z.unfold(-1, window, 1)
    left, singular, right = torch.linalg.svd(matrix, full_matrices=False)
    threshold = (shrink_scale*math.sqrt(2)*(math.sqrt(columns)+math.sqrt(window))
                 *observation_variance.sqrt())[:, None]
    reduction = (singular-threshold).clamp_min(0)-singular
    delta = (left*reduction[:, None, :])@right
    index = (torch.arange(columns, device=initial.device)[:, None]
             +torch.arange(window, device=initial.device)[None, :]).reshape(1, -1)
    output = torch.zeros_like(z).scatter_add_(1, index.expand(len(z), -1), delta.reshape(len(z), -1))
    count = torch.bincount(index.reshape(-1), minlength=256).to(initial)
    output = output/count[None]
    result = initial+mix*torch.stack((output.real, output.imag), dim=1)
    if not torch.isfinite(result).all():
        raise FloatingPointError('Finite Hankel reconstruction is required')
    return result
