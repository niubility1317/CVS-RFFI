"""Source-only loss emphasizing large observable physical innovations."""
import torch

# Source-only 75th percentile over 4096 fixed fit IDs and three views; seed 2026110803.
THRESHOLD = 0.00937667116522789
WEIGHT = .5


def observable_risk_penalty(output, anchor, innovation, clean, *, return_fraction=False):
    if (output.ndim != 3 or output.shape[1:] != (2, 256) or
            anchor.shape != output.shape or innovation.shape != output.shape or clean.shape != output.shape):
        raise ValueError('Matching source IQ and physical context required')
    anchor_energy = anchor.square().mean((1, 2)).clamp_min(1e-12)
    observable = innovation.square().mean((1, 2)) / anchor_energy
    selected = (observable >= THRESHOLD).to(output.dtype)
    clean_energy = clean.square().mean((1, 2)).clamp_min(1e-8)
    error = (output - clean).square().mean((1, 2)) / clean_energy
    loss = WEIGHT * (error * selected).sum() / selected.sum().clamp_min(1.)
    return (loss, selected.mean()) if return_fraction else loss
