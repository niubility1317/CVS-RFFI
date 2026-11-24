"""Separate the clean carrier from supervision of weak LoRa IQ corrections."""

import torch

from .angular_transport import AngularTransport
from .band_allocation import band_masks, fft, ifft
from .channel_blind_source import waveform_risk
from .common_anchor import iq_field
from .dura_repair import complex_iq, real_iq
from .joint_spectrum import split
from .joint_unroll_source import coordinates
from .natural_detail import paired_inputs, transfer_risk, weak_difference
from .split_shrink import spectrum, wave


TURN_LIMIT = .25


def physical_inner(left, right, rms):
    return ((left * right.conj()).real * rms.square()).sum((-2, -1), keepdim=True)


def limited_turn(anchor, proposed, rms, bands):
    """Match the physical size in each weak band and keep the remainder fixed."""
    base, alternative = fft(anchor), fft(proposed)
    correction = base * ~(bands[0] | bands[1])
    for mask in bands[:2]:
        reference = base * mask
        proposal = alternative * mask
        energy = physical_inner(reference, reference, rms)
        radial = physical_inner(proposal, reference, rms) / energy.clamp_min(1e-10)
        tangent = proposal - radial * reference
        tangent_energy = physical_inner(tangent, tangent, rms)
        cap = (TURN_LIMIT**2 * energy / tangent_energy.clamp_min(1e-10)).clamp(1e-12, 1.)
        direction = reference + cap.sqrt() * tangent
        turn_energy = physical_inner(direction, direction, rms)
        scale = (energy / turn_energy.clamp_min(1e-10)).clamp(1e-12, 1.)
        correction = correction + torch.where(energy > 1e-10, scale.sqrt() * direction, reference)
    return ifft(correction)


def source_reference(source, source_frame, tone, received, received_frame, *, separated):
    if not separated:
        return source / received_frame
    # Keep the received DC as the output coordinate anchor. Do not move the
    # much stronger clean tone through the independently estimated RX frame.
    weak = (source - tone * source_frame) / received_frame
    return weak - weak.mean(-1, keepdim=True) + received.mean(-1, keepdim=True)


class WeakSideband(AngularTransport):
    def __init__(self, separated):
        super().__init__(angular=True)
        self.separated = separated
        self.register_buffer('weak_bands', band_masks()[:2], persistent=False)

    def stages(self, y, rms, observed, *, use_action=True):
        anchor, _, scale, fields, action, _, _, _ = observed

        def constrained(logits):
            real, imag = logits.tanh().unbind(-2)
            proposal = wave(torch.complex(real, imag) * scale)
            return limited_turn(anchor, proposal, rms, self.weak_bands)

        coarse = self.read(self.coarse, fields)
        first = constrained(coarse)
        innovation = action.conj() * (y - action * (y + first))
        if not use_action:
            innovation = torch.zeros_like(innovation)
        refined = (*fields, *iq_field(spectrum(first), spectrum(y+first), spectrum(innovation)))
        return first, constrained(coarse + self.read(self.fine, refined))


def objective(model, tracker, iq, rng, *, with_identity, recipe, step):
    source = complex_iq(iq)
    clean, clean_phase, _, clean_scale = coordinates(tracker, iq)
    clean_frame = clean_phase * clean_scale
    delta = weak_difference(clean)
    weak = delta * clean_frame
    views = paired_inputs(source, recipe.perturb, rng, weak, with_identity=with_identity)
    observed, phase, quality, rms = coordinates(tracker, real_iq(views.flatten(0, 1)))
    frame = phase * rms
    target = torch.stack((source, source, source + weak), dim=1).flatten(0, 1)
    ref_frame = clean_frame.repeat_interleave(3, dim=0)
    ref_tone = clean.mean(-1, keepdim=True).repeat_interleave(3, dim=0)
    reference = source_reference(target, ref_frame, ref_tone, observed, frame, separated=model.separated)

    state = model.observe(observed, quality, rms)
    first, final = model.stages(observed, rms, state)
    first_recovered, final_recovered = observed + first, observed + final
    action, uncertainty = state[4:6]
    residual = ((observed - action*final_recovered).abs().square() / uncertainty).mean((1,2))
    forward = (residual-1.).clamp_min(0.).square().mean()
    output = (final_recovered * frame).unflatten(0, views.shape[:2])
    phase_anchor = (observed.mean(-1, keepdim=True) * frame).unflatten(0, views.shape[:2])
    detail = output - phase_anchor
    consistency = ((detail[:,0] - detail[:,1]) / clean_frame).abs().square().mean()
    transfer = split((output[:,2] - output[:,0]) / clean_frame)
    expected = split(delta)
    weak_common = transfer_risk(transfer[:,:1], expected[:,:1])
    weak_private = transfer_risk(transfer[:,1:], expected[:,1:])
    first_risk = waveform_risk(first_recovered, reference)
    final_risk = waveform_risk(final_recovered, reference)
    values = torch.stack((first_risk, final_risk, consistency, weak_common, weak_private, forward))
    a, b, c = recipe.loss_weights(step)
    return values @ values.new_tensor((a, b, c, .15, .35, .02)), values
