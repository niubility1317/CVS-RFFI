"""Joint waveform experts and an observation-only WiSig recovery selector."""

import torch
from torch import nn
from torch.nn import functional as F

from .observable_action import ObservableAction, generic_observation, weak_probe
from .phase_magnitude import magnitude_with_phase, match_power, weighted_spectrum
from .receiver_pilot import BEGIN, PERIOD, estimate, power


PROPOSALS = ('raw', 'coarse', 'fine', 'anchored', 'phase', 'amplitude')


def waveform_risk(proposals, reference):
    target = reference[:, None]
    energy = power(target).sum(dim=-1).clamp_min(1e-8)
    iq = power(proposals - target).sum(dim=-1) / energy
    amplitudes = torch.fft.fft(proposals, norm='ortho').abs()
    desired = torch.fft.fft(target, norm='ortho').abs()
    spectral = (amplitudes - desired).square().sum(dim=-1) / energy
    return iq.log1p() + .25 * spectral.log1p()


def proposal_bank(received, coarse, fine, anchored, noise):
    y = torch.fft.fft(received, norm='ortho')
    x = torch.fft.fft(fine, norm='ortho')
    corrected_phase = weighted_spectrum(y, match_power(y, x), noise)
    corrected_amplitude = magnitude_with_phase(x, y)
    return torch.stack((received, coarse, fine, anchored, torch.fft.ifft(corrected_phase, norm='ortho'),
                        torch.fft.ifft(corrected_amplitude, norm='ortho')), dim=1)


def repeat_error(aligned, start):
    first, second = aligned[:, start:start + PERIOD], aligned[:, start + PERIOD:start + 2 * PERIOD]
    return power(first - second).sum(-1) / (power(first).sum(-1) + power(second).sum(-1)).clamp_min(1e-8)


def choice_features(received, obs, candidates, info):
    common = [obs.coherence, obs.disturbance.log1p(), obs.slope * PERIOD,
              info['held_residual'].log1p(), info['action_delay'].float() / 40,
              info['action_gain'].log1p(), info['image_gain'].log1p()]
    common += [repeat_error(obs.aligned, t) for t in (BEGIN, 120, 160, 200)]
    energy = power(received).sum(-1, keepdim=True).clamp_min(1e-8)
    edit = power(candidates-received[:, None]).sum(-1) / energy
    magnitude = power(candidates).sum(-1) / energy
    y_fft = torch.fft.fft(received, norm='ortho').abs()
    candidate_fft = torch.fft.fft(candidates, norm='ortho').abs()
    envelope_edit = (candidate_fft-y_fft[:, None]).square().sum(-1) / energy
    return torch.cat((torch.stack(common, dim=1),
                      edit.log1p(), magnitude.log1p(), envelope_edit.log1p()), dim=1)


class WaveformChoice(ObservableAction):
    def __init__(self):
        super().__init__(fitted=True, protect_amplitude=True)
        self.selector = nn.Sequential(nn.Linear(11 + 3*len(PROPOSALS), 48), nn.SiLU(),
                                      nn.Linear(48, len(PROPOSALS)))
        with torch.no_grad():
            self.selector[-1].bias.zero_()
            self.selector[-1].bias[0] = .30

    def forward(self, received):
        coarse, anchored, info, fine = super().forward(received, return_unprotected=True)
        obs = estimate(received)
        candidates = proposal_bank(received, coarse, fine, anchored, obs.noise_power)
        # Quality evidence is measured on the proposals. Gradients from the
        # proposed waveforms cannot reduce the ranking loss via these inputs.
        features = choice_features(received, obs, candidates, info).detach()
        return candidates, self.selector(features)


def select(candidates, logits, minimum_margin=None):
    confidence, index = logits[:, 1:].max(dim=1)
    margin = confidence - logits[:, 0]
    threshold = 0. if minimum_margin is None else minimum_margin
    selected = torch.where(margin > threshold, index + 1, 0)
    row = torch.arange(len(candidates), device=candidates.device)
    return candidates[row, selected], selected, margin


def objective(model, clean, generator):
    delta = weak_probe(clean)
    target = torch.stack((clean, clean, clean + delta), dim=1)
    pair = generic_observation(target[:, (0, 2)], generator)
    independent = generic_observation(target[:, 1:2], generator)
    observed = torch.stack((pair[:, 0], independent[:, 0], pair[:, 1]), dim=1)
    observed[-4:] = target[-4:]
    candidates, logits = model(observed.flatten(0, 1))
    risks = waveform_risk(candidates, target.flatten(0, 1))
    actual = (-risks.detach() / .20).softmax(dim=1)
    ranking = -(actual * F.log_softmax(logits, dim=1)).sum(1).mean()
    expected = (logits.softmax(dim=1) * risks).sum(1).mean()
    health = risks[:, 1:].mean()
    proposals = candidates[:, 1:].unflatten(0, target.shape[:2])
    transported = proposals[:, 2] - proposals[:, 0]
    error = power(transported - delta[:, None]).mean(-1)
    transfer = (error / (.002 + power(delta).mean(-1, keepdim=True))).log1p().mean()
    overlap = (power(proposals[:, 1]-proposals[:, 0]).mean(-1)
               / power(clean).mean(-1, keepdim=True).clamp_min(1e-8)).log1p().mean()
    values = torch.stack((health, expected, ranking, transfer, overlap))
    return (values * values.new_tensor((.43, .75, .25, .35, .08))).sum(), values
