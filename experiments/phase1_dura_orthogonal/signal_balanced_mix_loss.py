"""Receiver-balanced source-only learned mixing objective."""
import numpy as np
import torch


def receiver_weights(rx):
    rx = np.asarray(rx)
    groups, inverse, counts = np.unique(rx, return_inverse=True, return_counts=True)
    if rx.ndim != 1 or not len(rx) or len(groups) != 5:
        raise ValueError('Five nonempty source receiver groups required')
    return torch.as_tensor((len(rx) / (len(groups) * counts[inverse])).astype(np.float32))


def balanced_oracle_loss(output, anchor, proposal, clean, weight):
    if output.shape != anchor.shape or output.shape != proposal.shape or output.shape != clean.shape:
        raise ValueError('Matching source IQ shapes required')
    energy = clean.square().mean((1, 2)).clamp_min(1e-8)
    delta = proposal - anchor
    q = delta.square().mean((1, 2))
    valid = q > 1e-10 * energy
    alpha = ((output-anchor)*delta).mean((1, 2)) / q.clamp_min(1e-12)
    oracle = (((clean-anchor)*delta).mean((1, 2)) / q.clamp_min(1e-12)).clamp(0, 1).detach()
    iq = (output-clean).square().mean((1, 2))/energy
    change = (output-anchor).square().mean((1, 2))/energy
    distill = (alpha-oracle).square() * valid
    return (weight*(iq + .02*change + .02*distill)).sum()/weight.sum()
