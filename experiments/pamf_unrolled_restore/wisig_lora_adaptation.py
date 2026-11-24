"""WiSig adaptations of the source-trained LoRa restoration structures."""

import torch

from .wisig_rank4_observation import posterior_increment
from .wisig_receiver_estimates import receive_coordinates


def as_iq(value):
    return torch.stack((value.real, value.imag), dim=1)


def expanded_readout(model, iq):
    y, level, quality, variance = receive_coordinates(iq)
    repeated = y[:, None, :].expand(-1, 6, -1)
    qualities = quality[:, None, :].expand(-1, 6, -1)
    estimates = variance[:, None, :].expand(-1, 6, -1)
    stages, *_ = model.stages(repeated, qualities, level[:, None, :], receiver_variance=estimates)
    # All six tensors represent the same WiSig observation, never six packets.
    final = stages[-1].mean(dim=1)
    return as_iq((y + .5 * (final - y)) * level)


def rank_readout(model, iq, diagonal, factors):
    y, level, quality, variance = receive_coordinates(iq)
    mixed = model.training and y.is_cuda and torch.cuda.is_bf16_supported()
    _, proposal, _ = model.propose(y[:, None], quality[:, None], mixed_conv=mixed)
    measured = torch.fft.fft(y, norm='ortho')
    center = torch.fft.fft(proposal[:, 0], norm='ortho')
    residual = measured - center
    coordinates = torch.cat((residual.real, residual.imag), dim=-1)
    updated = posterior_increment(coordinates, diagonal, factors, (variance * .5).clamp_min(1e-6))
    real, imag = updated.chunk(2, dim=-1)
    recovered = torch.fft.ifft(center + torch.complex(real, imag), norm='ortho')
    return as_iq((y + .5 * (recovered - y)) * level)
