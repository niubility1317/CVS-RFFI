"""Project a proposed IQ edit onto reliable repeated preamble structure."""

import torch


# LoRa 1 MS/s native upchirps and the audited WiSig 25 MS/s L-STF window.
PREAMBLE = {8192: (0, 8, 1024), 256: (80, 4, 20)}
MAX_EDIT = .25


def repeat_rotation(blocks):
    adjacent = (blocks[:, 1:] * blocks[:, :-1].conj()).sum(-1)
    rotation = torch.polar(torch.ones_like(adjacent.real), adjacent.angle())
    start = torch.ones_like(rotation[:, :1])
    return torch.cat((start, rotation), dim=1).cumprod(1)


def coherent_edit(blocks, edit):
    rotation = repeat_rotation(blocks)[..., None]
    observed = torch.fft.fft(blocks * rotation.conj(), norm='ortho')
    edit_spectrum = torch.fft.fft(edit * rotation.conj(), norm='ortho')
    average = observed.mean(1, keepdim=True)
    noise = (observed - average).abs().square().sum(1, keepdim=True) / (blocks.shape[1] - 1)
    signal = (average.abs().square() - noise / blocks.shape[1]).clamp_min(0.)
    confidence = signal / (signal + noise).clamp_min(1e-12)
    common = confidence * edit_spectrum.mean(1, keepdim=True)
    return torch.fft.ifft(common, norm='ortho') * rotation


def repeated_edit_outputs(received, proposal):
    """Return coherent and unprojected edits with the same energy and support."""
    if (received.ndim != 2 or not received.is_complex() or proposal.shape != received.shape
            or not proposal.is_complex() or received.shape[-1] not in PREAMBLE):
        raise ValueError('Expected matching received/proposed native complex IQ')
    start, count, period = PREAMBLE[received.shape[-1]]
    stop = start + count * period
    segment = received[:, start:stop]
    edits = proposal[:, start:stop] - segment
    projected = coherent_edit(segment.unflatten(-1, (count, period)),
                              edits.unflatten(-1, (count, period))).flatten(1)
    projected_norm = projected.norm(dim=-1, keepdim=True)
    edit_norm = edits.norm(dim=-1, keepdim=True)
    bound = MAX_EDIT * segment.norm(dim=-1, keepdim=True)
    dose = torch.minimum(projected_norm, torch.minimum(edit_norm, bound))
    pair = (projected * (dose / projected_norm.clamp_min(1e-12)),
            edits * (dose / edit_norm.clamp_min(1e-12)))
    return tuple(torch.cat((received[:, :start], segment + delta, received[:, stop:]), dim=-1)
                 for delta in pair)
