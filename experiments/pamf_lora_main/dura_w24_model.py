"""Eight-chirp DURA with the W24 receive aperture and edge-conditioned priors."""

import numpy as np
from scipy.signal import firwin
import torch

from experiments.cric_dura_comparison.io import tensor
from .channel_blind_source import waveform_risk
from .data import WINDOW_START
from .dura_capacity import DuraCapacity
from .dura_repair import PacketRestorer, complex_iq, real_iq
from .dura_w24 import EDGES, STARTS, WIDE
from .joint_spectrum import split
from .joint_unroll_source import coordinates
from .natural_detail import transfer_risk, weak_difference
from .packet_data import PACKET_LEN
from .posterior_transport import condition_waveforms


def front_end():
    phase = np.arange(WIDE) * 125
    sample, remainder = np.divmod(phase, 48)
    offsets = np.arange(-26, 28)
    taps = 48 * firwin(2501, 1/125, window=('kaiser', 5.0)).astype(np.float32)
    delta = remainder[:, None] - 48 * offsets
    weights = np.where(np.abs(delta) <= 1250, taps[(delta + 1250).clip(0, 2500)], 0.)
    index = STARTS[:, None, None] + WINDOW_START + sample[:, None] + offsets
    index = np.where(index < 0, -index, index)
    index = np.where(index >= PACKET_LEN, 2 * PACKET_LEN - 2 - index, index)
    return index, weights


class WideTracker(PacketRestorer):
    def __init__(self):
        super().__init__('consensus', samples=WIDE)
        index, taps = front_end()
        self.register_buffer('indices', tensor(index), persistent=False)
        self.register_buffer('taps', tensor(taps), persistent=False)
        self.register_buffer('edges', tensor(EDGES), persistent=False)

    def aperture(self, raw):
        core = (raw[..., self.indices] * self.taps).sum(-1)
        return torch.cat((core, raw[..., self.edges]), dim=-1)


class DuraW24(DuraCapacity):
    def __init__(self):
        super().__init__('expanded', samples=WIDE, condition_fields=6)

    def propose(self, y, quality, rms, edges):
        edge_pairs = edges.unflatten(-1, (2, 24))
        power = edge_pairs.abs().square().mean(-1).clamp_min(1e-10)
        step = (edge_pairs[..., 1:] * edge_pairs[..., :-1].conj()).mean(-1) / power
        log_power = (power/rms.square()).log().clamp(-4., 4.)
        fields = tuple(x.unsqueeze(-1) for t in (log_power, step.real, step.imag) for x in t.unbind(-1))
        return self.stages(y, quality, rms, condition=fields)

    def score(self, tracker, observed):
        y = coordinates(tracker, observed[..., :WIDE])[0]
        outer = torch.fft.fft(y, norm='ortho')[..., self.noise_band].abs().square()
        return (outer.median(-1).values/.69314718056).sqrt().median(1).values

    def restore(self, tracker, observed, threshold):
        mask = self.score(tracker, observed) > threshold
        if not mask.any():
            return observed, mask
        selected = observed[mask]
        y, phase, quality, rms = coordinates(tracker, selected[..., :WIDE])
        stages, *_ = self.propose(y, quality, rms, complex_iq(selected[..., WIDE:]))
        out = observed.clone()
        out[mask, :, :, :WIDE] += real_iq(.5 * (stages[-1]-y) * phase * rms)
        return out, mask


@torch.no_grad()
def calibrate(model, tracker, packets):
    scores = [model.score(tracker, real_iq(tracker.aperture(raw))) for raw in packets.split(32)]
    threshold = float(1.05 * torch.cat(scores).max())
    return threshold, {'gate_calibration': 'clean W24 outer complex RMS',
                       'parameters': sum(p.numel() for p in model.parameters())}


def objective(model, tracker, raw, rng, *, with_identity, recipe, step):
    raw = raw / raw.abs().square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
    clean = tracker.aperture(raw)
    center = clean[..., :WIDE]
    framed, phase, _, rms = coordinates(tracker, real_iq(center))
    clean_frame = phase*rms
    delta = weak_difference(framed)
    change = delta*clean_frame
    # Add generic disturbances at capture rate before making either view.
    a, b = (tracker.aperture(condition_waveforms(raw.unflatten(-1, (8, 1024)), rng).flatten(-2))
            for _ in range(2))
    if with_identity:
        a[-2:] = b[-2:] = clean[-2:]
    data = torch.stack((a, b, torch.cat((a[..., :WIDE]+change, a[..., WIDE:]), dim=-1)), dim=1)
    observed = data.flatten(0, 1)
    y, phase, quality, rms = coordinates(tracker, real_iq(observed[..., :WIDE]))
    frame = phase*rms
    ground = torch.stack((center, center, center+change), dim=1).flatten(0, 1)/frame
    (coarse, first, final), action, variance, _, _ = model.propose(y, quality, rms, observed[..., WIDE:])
    half = y + .5*(final-y)
    prediction = (half*frame).unflatten(0, data.shape[:2])
    corruption = (y-action(ground)).abs().square()
    signal = split((prediction[:, 2]-prediction[:, 0])/clean_frame)
    expected = split(delta)
    preserve = transfer_risk(signal[:, :1], expected[:, :1]) + transfer_risk(signal[:, 1:], expected[:, 1:])
    consistency = ((prediction[:, 0]-prediction[:, 1])/clean_frame).abs().square().mean()
    identity = ((prediction-data[..., :WIDE])[-2:, :2].abs().square().mean()
                if with_identity else y.real.new_zeros(()))
    values = torch.stack((waveform_risk(coarse, ground), waveform_risk(first, ground),
                          waveform_risk(final, ground), waveform_risk(half, ground), preserve, consistency,
                          (corruption/(.01+variance.detach())).log1p().mean(),
                          (corruption.detach()/variance + variance.log()).mean(), identity))
    a_weight, b_weight, c_weight = recipe.loss_weights(step)
    weights = values.new_tensor((.6*a_weight, .4*a_weight, .35*b_weight, .65*b_weight,
                                 .30, c_weight, .07, .015, .35))
    return values @ weights, values
