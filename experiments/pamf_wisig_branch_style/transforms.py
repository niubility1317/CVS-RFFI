"""Branch-local transformations with detached source donor statistics."""

import math

import torch
from torch.nn import functional as F

from experiments.pamf_ssl_research.u_mixstyle_objective import mix_moments


def moment_mix(module, feature, partner, amount, valid):
    return mix_moments(feature, partner, valid, 1 - amount, 1e-6)


def frequency_envelope(module, feature, partner, amount, valid):
    # Orthonormal DC and first two cosine modes on the compressed band axis.
    bins = feature.shape[-1]
    position = (torch.arange(bins, device=feature.device, dtype=feature.dtype) + .5) / bins
    basis = torch.stack((torch.ones_like(position) / math.sqrt(bins),
                         torch.cos(math.pi * position) * math.sqrt(2 / bins),
                         torch.cos(2 * math.pi * position) * math.sqrt(2 / bins)))
    source = feature.detach()
    coefficients = source @ basis.T
    shift = (coefficients[partner] - coefficients) @ basis
    cap = .25 * source.square().mean(-1, keepdim=True).sqrt()
    scale = (cap / shift.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)).clamp_max(1)
    return torch.where(valid[:, None, None], feature + amount * shift * scale, feature)


def pa_envelope(module, feature, partner, amount, valid):
    # Order 1 at lag zero supplies the envelope. Every I/Q and odd-order pair
    # for the same lag receives the same gain at the same input index.
    length = feature.shape[-1]
    depth, orders = module.memory_depth, module.orders
    if orders[0] != 1 or feature.shape[1] != 2 * depth * len(orders) or length != 256:
        raise ValueError("Expected the native PA lift with its first order and lag axes")
    power = feature[:, :2].detach().square().sum(1).reshape(-1, 8, 32).mean(-1)
    log_rms = .5 * power.clamp_min(1e-6).log()
    log_gain = amount * (log_rms[partner] - log_rms)[:, None, :].clamp(-.22, .22)
    gain = F.interpolate(log_gain, size=length, mode="linear", align_corners=False).exp()
    lag_gain = F.pad(gain, (depth - 1, 0), value=1).unfold(-1, length, 1).flip(-2)
    mixed = (feature.reshape(-1, depth, 2 * len(orders), length)
             * lag_gain.transpose(1, 2)).reshape_as(feature)
    return torch.where(valid[:, None, None], mixed, feature)


def reference_trend(module, feature, partner, amount, valid):
    if feature.shape[-1] != 29:
        raise ValueError("Expected 12 complex reference residuals and five auxiliary coordinates")
    coordinates = feature.new_tensor([-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6]) / 6
    square = coordinates.square() - coordinates.square().mean()
    basis = F.normalize(torch.stack((torch.ones_like(coordinates), coordinates, square)), dim=1)
    response = feature[:, :24].reshape(-1, 2, 12)
    source = response.detach()
    coefficients = source @ basis.T
    shift = (coefficients[partner] - coefficients) @ basis
    # One scale per packet keeps the change within the smooth subspace.
    peak = shift.square().sum(1).sqrt().amax(1, keepdim=True).clamp_min(1e-6)
    headroom = (4 - source.square().sum(1).sqrt().amax(1, keepdim=True)).clamp_min(0)
    cap = headroom.clamp_max(.15)
    mixed = response + amount * shift * (cap / peak).clamp_max(1)[:, :, None]
    mixed = torch.where(valid[:, None, None], mixed, response)
    return torch.cat((mixed.flatten(1), feature[:, 24:]), dim=1)


def signature_bend(module, feature, partner, amount, valid):
    if feature.shape[1:] != (6, 32):
        raise ValueError("Expected the native six-coordinate, four-window signature projection")
    windows = feature.reshape(-1, 6, 4, 8)
    phase = torch.linspace(0, 1, 8, device=feature.device, dtype=feature.dtype)
    chord = windows[..., :1] * (1 - phase) + windows[..., -1:] * phase
    bend = windows - chord
    energy = bend.detach().square().mean(dim=(1, 3), keepdim=True)
    log_rms = .5 * energy.clamp_min(1e-6).log()
    scale_delta = (log_rms[partner] - log_rms).clamp(-.22, .22)
    scale = (amount.unsqueeze(-1) * scale_delta).exp()
    interior = windows[..., 1:-1] + (scale - 1) * bend[..., 1:-1]
    mixed = torch.cat((windows[..., :1], interior, windows[..., -1:]), dim=-1)
    return torch.where(valid[:, None, None], mixed.reshape_as(feature), feature)
