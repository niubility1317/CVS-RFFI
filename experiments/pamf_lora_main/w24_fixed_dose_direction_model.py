"""Learn a packet edit direction at the frozen W24 center and edge amplitudes."""

import torch
from torch import nn

from .w24_native_center import CENTER, NATIVE_CHIRP, NATIVE_LENGTH, NativeCenterProjection
from .w24_packet_dura import PacketPrior, fields


EDGE = 24


class DirectionHead(nn.Module):
    """Refine the direction of the frozen half edit using the entire capture."""

    def __init__(self):
        super().__init__()
        self.prior = PacketPrior(8, 2)

    def forward(self, observed, base_edit):
        if (observed.ndim != 2 or observed.shape[-1] != NATIVE_LENGTH or not observed.is_complex()
                or base_edit.shape != observed.shape or base_edit.dtype != observed.dtype):
            raise ValueError('Expected matching native complex observations and edits')
        y, edit = (v.unflatten(-1, (-1, NATIVE_CHIRP)) for v in (observed, base_edit))
        scale = y.abs().square().mean((-2, -1), keepdim=True).clamp_min(1e-12).sqrt()
        normalized, update = y / scale, edit / scale
        real, imag = self.prior(*fields(normalized, update,
                                        normalized - normalized.mean(-2, keepdim=True),
                                        update - update.mean(-2, keepdim=True)))
        edit_rms = torch.linalg.vector_norm(edit, dim=(-2, -1), keepdim=True) / (NATIVE_LENGTH**.5)
        return (edit + torch.complex(real, imag) * edit_rms).flatten(-2)


def _norm(wave):
    flat = wave.flatten(1)
    shape = (-1,) + (1,)*(wave.ndim-1)
    return torch.linalg.vector_norm(flat, dim=1).reshape(shape)


def _rescale_edit(base, proposal):
    want, got = _norm(base), _norm(proposal)
    tiny = torch.finfo(got.dtype).tiny
    # A zero proposal has no direction. Keep the original edit in that case.
    return torch.where(got > tiny, proposal / got.clamp_min(tiny) * want, base)


def matched_aperture_edit(base_aperture_edit, proposal_aperture_edit):
    """Keep the original packet L2 norms separately in center and edge regions."""
    return torch.cat(tuple(
        _rescale_edit(base_aperture_edit[..., region], proposal_aperture_edit[..., region])
        for region in (slice(None, CENTER), slice(CENTER, None))
    ), dim=-1)


class ApertureDirection(nn.Module):
    """Apply the native W24 projection before the two fixed-dose constraints."""

    def __init__(self):
        super().__init__()
        self.project = NativeCenterProjection()

    def project_regions(self, wave):
        native = wave.unflatten(-1, (-1, NATIVE_CHIRP))
        return torch.cat((self.project(wave), native[..., :EDGE], native[..., -EDGE:]), dim=-1)

    def edits(self, base_edit, proposal):
        base, predicted = self.project_regions(torch.cat((base_edit, proposal))).chunk(2)
        return base, matched_aperture_edit(base, predicted)

    @staticmethod
    def normalize(aperture):
        rms = aperture[..., :CENTER].abs().square().mean((-2, -1), keepdim=True).clamp_min(1e-24).sqrt()
        return torch.view_as_real(aperture / rms).transpose(-1, -2).contiguous()

    def forward(self, observed, base_edit, proposal):
        direct = self.project_regions(observed)
        base, refined = self.edits(base_edit, proposal)
        return tuple(self.normalize(view) for view in (direct, direct + base, direct + refined))


def _masked_mean(value, mask):
    return (value * mask).sum() / mask.sum().clamp_min(1)


def _unit(wave):
    norm = _norm(wave)
    return wave / norm.clamp_min(1e-12), norm.flatten() > 1e-12


def _angle(prediction, reference, use):
    direction, prediction_valid = _unit(prediction)
    target, reference_valid = _unit(reference)
    valid = use & prediction_valid & reference_valid
    cosine = (direction.conj() * target).real.flatten(1).sum(-1).clamp(-1., 1.)
    return _masked_mean(1. - cosine, valid), _masked_mean(cosine, valid), direction - target, valid


def direction_objective(aperture, observed, base_edit, proposal, noiseless, *, clean_mask=None):
    """Use simulator components only as source loss targets for direction and identity."""
    oracle = noiseless - observed
    if clean_mask is None:
        clean_mask = ~_unit(oracle)[1]
    noisy = ~clean_mask
    base, predicted = aperture.edits(base_edit, proposal)
    target = aperture.project_regions(oracle)

    native_angle, native_cosine, _, _ = _angle(proposal, oracle, noisy)
    center_angle, center_cosine, center_error, center_valid = _angle(
        predicted[..., :CENTER], target[..., :CENTER], noisy)
    edge_angle, edge_cosine, _, _ = _angle(predicted[..., CENTER:], target[..., CENTER:], noisy)
    with torch.no_grad():
        signal = aperture.project(noiseless)
        power = torch.fft.fft(signal, norm='ortho').abs().square()
        spectral_weight = (1. + power / power.mean(-1, keepdim=True).clamp_min(1e-12)).reciprocal()
    spectral_error = torch.fft.fft(center_error, norm='ortho').abs().square()
    weak_spectrum = _masked_mean((spectral_weight * spectral_error).sum((-2, -1)), center_valid)

    clean_center, _, _, _ = _angle(predicted[..., :CENTER], base[..., :CENTER], clean_mask)
    clean_edges, _, _, _ = _angle(predicted[..., CENTER:], base[..., CENTER:], clean_mask)
    identity = clean_center + .35 * clean_edges
    loss = center_angle + .35 * edge_angle + .25 * native_angle + .15 * weak_spectrum + .50 * identity
    metrics = dict(center_cosine=center_cosine, edge_cosine=edge_cosine, native_cosine=native_cosine,
                   weak_spectrum=weak_spectrum, identity=identity)
    return loss, {key: value.detach() for key, value in metrics.items()}
