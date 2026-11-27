"""Orthogonal center components shared by source training and received-IQ readout."""

import torch


def aligned_spectrum_parts(value, phase):
    """Split the measured-phase spectrum into common and zero-sum chirp parts."""
    frame = phase / phase.abs().clamp_min(1e-12)
    spec = torch.fft.fft(value * frame.conj(), norm='ortho')
    n = value.shape[-1]
    near = (torch.fft.fftfreq(n, device=value.device) * n).round().abs() <= 32
    common = spec.mean(-2, keepdim=True)
    return spec, common.expand_as(spec), spec - common, near, frame


def components(value, phase):
    _, common, private, near, frame = aligned_spectrum_parts(value, phase)
    return torch.stack((common * near, common * ~near, private * near, private * ~near), dim=1), frame


def component_norm(value):
    return torch.linalg.vector_norm(value, dim=(-2, -1), keepdim=True)


def matched_component_edit(base, proposal, phase):
    """Keep each estimated parent norm while replacing its within-component direction."""
    baseline, frame = components(base, phase)
    candidate, _ = components(proposal, phase)
    want, got = (component_norm(value) for value in (baseline, candidate))
    floor = (component_norm(proposal).unsqueeze(1) * 1e-6).clamp_min(1e-12)
    shaped = torch.where(got > floor, candidate / got.clamp_min(1e-12) * want, baseline)
    return torch.fft.ifft(shaped.sum(1), norm='ortho') * frame


def component_angle(proposal, target, phase):
    """Measure four directions against legal same-packet source supervision."""
    estimate, _ = components(proposal, phase)
    reference, _ = components(target, phase)
    got, want = (component_norm(value) for value in (estimate, reference))
    floor = (component_norm(target).unsqueeze(1) * 1e-6).clamp_min(1e-12)
    valid = (want > floor) & (got > 1e-12)
    cosine = ((estimate / got.clamp_min(1e-12)).conj() * (reference / want.clamp_min(1e-12))
              ).real.sum((-2, -1), keepdim=True).clamp(-1., 1.)
    loss = ((1. - cosine) * valid).sum() / valid.sum().clamp_min(1)
    per_component = (cosine * valid).sum(0).flatten() / valid.sum(0).flatten().clamp_min(1)
    return loss, per_component.detach()
