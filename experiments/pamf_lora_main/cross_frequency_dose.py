"""Learn continuous readout strengths from a fixed LoRa covariance direction."""

import torch
from torch import nn

from .channel_blind_source import waveform_risk
from .cross_frequency import CrossFrequencyPrior
from .dura_repair import complex_iq, real_iq
from .joint_spectrum import limit, split
from .joint_unroll_source import coordinates
from .natural_detail import paired_inputs, transfer_risk, weak_difference
from .received_update import ReceivedUpdate
from .variance_transport import physical_norm, posterior_gain


def head(inputs):
    net = nn.Sequential(nn.Linear(inputs, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU(), nn.Linear(32, 2))
    nn.init.normal_(net[-1].weight, std=.01)
    nn.init.zeros_(net[-1].bias)
    return net


class CrossFrequencyDose(nn.Module):
    def __init__(self, separate):
        super().__init__()
        self.separate = separate
        self.prior = CrossFrequencyPrior()
        self.coarse = head(15)
        self.fine = head(21)

    def directions(self, spectrum, noise):
        values = torch.view_as_real(spectrum).flatten(-2) - self.prior.center
        edits = []
        for index, component in enumerate((values[:, :1], values[:, 1:])):
            axes = self.prior.joint_axes[index]
            gain = posterior_gain(self.prior.joint_variance[index], noise[:, index, None, None])
            edits.append(((component @ axes) * (gain.sqrt() - 1.)) @ axes.T)
        packed = torch.view_as_complex(torch.cat(edits, dim=1).unflatten(-1, (255, 2)).contiguous())
        full = torch.cat((torch.zeros_like(packed[..., :1]), packed), dim=-1)
        change = self.prior.basis.mT.to(full) @ torch.fft.ifft(full, norm='ortho')
        common = change.mean(dim=1, keepdim=True).expand_as(change)
        return common, change - common

    @staticmethod
    def vector_features(y, q, noise, mismatch, action, rays, rms):
        even, odd = y[:, ::2].mean(1), y[:, 1::2].mean(1)
        power_even, power_odd = (wave.abs().square().mean(-1) for wave in (even, odd))
        overlap = (even * odd.conj()).real.mean(-1)
        coherence = overlap / (power_even * power_odd).clamp_min(1e-10).sqrt()
        even_odd = (even - odd).abs().square().mean(-1)
        common = y.mean(1, keepdim=True)
        common_power = common.abs().square().mean((1, 2))
        detail_power = (y - common).abs().square().mean((1, 2))
        raw_power = physical_norm(y, rms).flatten().clamp_min(1e-8)
        amounts = [physical_norm(ray, rms).flatten() / raw_power for ray in rays]
        signs = [(y.conj() * ray * rms.square()).real.mean((1, 2)) / raw_power.square() for ray in rays]
        return torch.stack((
            (10. * noise[:, 0]).log1p(), (10. * noise[:, 1]).log1p(),
            coherence, (5. * even_odd).log1p(),
            common_power, detail_power,
            q.mean((1, 2)), q.amin((1, 2)), (4. * mismatch.mean((1, 2))).log1p(),
            (action - 1.).abs().mean((1, 2)), (action - 1.).abs().amax((1, 2)),
            *amounts, *signs,
        ), dim=-1)

    @staticmethod
    def apply(y, rms, rays, doses):
        shared, private = doses[:, :, None, None].unbind(1)
        change = shared * rays[0] + private * rays[1]
        return limit(change, y, rms, y.real.new_tensor(.30))

    @staticmethod
    def dose(logits, separate):
        if not separate:
            logits = logits.mean(-1, keepdim=True).expand_as(logits)
        return 1.25 * logits.sigmoid()

    def observe(self, y, q, rms, *, use_action=True, use_noise=True):
        spectrum = self.prior.spectrum(y)
        measured = self.prior.noise(spectrum)
        noise = measured if use_noise else torch.full_like(measured, .05)
        action, reference, mismatch = ReceivedUpdate.estimate(y, noise.mean(-1)[:, None, None])
        if not use_action:
            action = torch.ones_like(action)
        rays = self.directions(spectrum, noise)
        fields = self.vector_features(y, q, noise, mismatch, action, rays, rms)
        uncertainty = .015 + 2. * noise.mean(-1)[:, None, None] + mismatch
        return fields, rays, action, reference, uncertainty, measured, mismatch

    def solve(self, y, rms, observed, *, separate=None):
        separated = self.separate if separate is None else separate
        features, rays, action, reference, uncertainty, _, _ = observed
        first_logits = self.coarse(features)
        initial = self.dose(first_logits, separated)
        first = self.apply(y, rms, rays, initial)
        mismatch = (y - action * (y + first)).abs().square() / uncertainty
        reference_hold = ((y + first - reference).abs().square()
                          - (y - reference).abs().square()) / uncertainty
        residual = torch.stack((mismatch.mean((1, 2)).log1p(),
                                mismatch.mean(-1).amax(-1).log1p(),
                                reference_hold.mean((1, 2)).clamp(-4, 4),
                                reference_hold.mean(-1).abs().mean(-1).log1p()), dim=1)
        fine_logits = first_logits + self.fine(torch.cat((features, initial, residual), dim=-1))
        doses = self.dose(fine_logits, separated)
        return first, self.apply(y, rms, rays, doses), doses


@torch.no_grad()
def initialize(model, tracker, training, _calibration):
    model.prior.calibrate(coordinates(tracker, batch)[0] for batch in training.split(96))


def objective(model, tracker, iq, rng, *, with_identity, recipe, step):
    clean, clean_rotation, _, clean_rms = coordinates(tracker, iq)
    clean_frame = clean_rotation * clean_rms
    source = complex_iq(iq)
    difference = weak_difference(clean)
    weak = difference * clean_frame
    views = paired_inputs(source, recipe.perturb, rng, weak, with_identity=with_identity)
    y, rotation, quality, rms = coordinates(tracker, real_iq(views.flatten(0, 1)))
    frame = rotation * rms
    target = torch.stack((source, source, source + weak), dim=1).flatten(0, 1) / frame
    observed = model.observe(y, quality, rms)
    coarse, fine, _ = model.solve(y, rms, observed)
    _, _, action, _, uncertainty, _, _ = observed
    explain = ((y - action * (y + fine)).abs().square() / uncertainty).mean((1, 2))
    observation = (explain - 1.).clamp_min(0.).square().mean()
    restored = ((y + fine) * frame).unflatten(0, views.shape[:2])
    consistency = ((restored[:, 0] - restored[:, 1]) / clean_frame).abs().square().mean()
    projected_difference = split((restored[:, 2] - restored[:, 0]) / clean_frame)
    expected = split(difference)
    shared = transfer_risk(projected_difference[:, :1], expected[:, :1])
    private = transfer_risk(projected_difference[:, 1:], expected[:, 1:])
    values = torch.stack((waveform_risk(y + coarse, target), waveform_risk(y + fine, target),
                          consistency, shared, private, observation))
    first, final, paired = recipe.loss_weights(step)
    total = first * values[0] + final * values[1] + paired * consistency + .21 * shared + .34 * private + .16 * observation
    return total, values
