"""Inject real spectral gains into each frozen R17 physical solve."""

from contextlib import nullcontext

import torch

from experiments.pamf_action_restore.models import SCALE
from experiments.pamf_restoration.models import complex_iq, pair
from experiments.pamf_uncertainty_restore.models import weighted_map


def fused_unroll(r17, gain, received, real_prior, *, train_action=False, return_action=False):
    if received.ndim != 2 or received.shape != real_prior.shape or received.shape[-1] != 256:
        raise ValueError('Expected matching complex WiSig packets')
    received_pair = pair(received)
    with nullcontext() if train_action else torch.no_grad():
        base, action, native = r17.frozen.estimate(received_pair)
    variance = native[:, None].expand(-1, 256)
    logvar = native.log().clamp(-16, 4)[:, None, None].expand(-1, 1, 256) / 8
    current = base
    for k in range(3):
        innovation = pair(action.adjoint(received - action(complex_iq(current))))
        innovation = innovation / (innovation.square().mean((1, 2), keepdim=True) + 1e-6).sqrt()
        inputs = torch.cat((received_pair, base, current, innovation, logvar), 1)
        proposal = base + SCALE * r17.prior[0 if len(r17.prior) == 1 else k](inputs)
        corrected = gain(received, complex_iq(proposal), real_prior, action)
        current = pair(weighted_map(action, received, corrected, variance, steps=2,
                                    initial=complex_iq(current)))
    output = complex_iq(current)
    return (output, action) if return_action else output


def fused_anchor_unroll(r17, gain, received, anchor):
    if received.ndim != 2 or received.shape != anchor.shape or received.shape[-1] != 8 * gain.width:
        raise ValueError('Expected matching complete packets for the gain width')
    scale = received.abs().square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
    observed = received / scale
    received_pair = pair(observed)
    initial_anchor = anchor / scale
    current_anchor = initial_anchor
    with torch.no_grad():
        base, action, native = r17.estimate(received_pair)
    variance = native[:, None].expand_as(observed.real)
    logvar = native.log().clamp(-16, 4)[:, None, None].expand(-1, 1, observed.shape[-1]) / 8
    current = base
    for core in r17.priors:
        innovation = pair(action.adjoint(observed - action(complex_iq(current))))
        innovation = innovation / (innovation.square().mean((1, 2), keepdim=True) + 1e-6).sqrt()
        inputs = torch.cat((received_pair, base, current, innovation, logvar), 1)
        proposal = base + SCALE * core(inputs) + pair(current_anchor - initial_anchor)
        current = pair(weighted_map(action, observed, complex_iq(proposal), variance, steps=2,
                                    initial=complex_iq(current)))
        current_anchor = gain(observed, current_anchor, complex_iq(current), action)
    return current_anchor * scale
