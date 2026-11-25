"""LoRa DURA with one receive-only rule on the recorded narrow and W24 grids."""

from dataclasses import replace

import torch
from torch import nn

from experiments.cric_dura_comparison.io import tensor
from .dura_capacity import DuraCapacity
from .dura_repair import PacketRestorer, complex_iq, real_iq
from .dura_w24 import EDGE, NARROW, WIDE
from .dura_w24_model import WideTracker, boundary_fields, front_end, objective as waveform_objective
from .dura_w24_train import RECIPE as W24_RECIPE
from .joint_unroll_source import coordinates


def shared_guard_coefficients(value):
    """Per-bin amplitude on a 384 kS/s reference, measured in each received grid."""
    n = value.shape[-1]
    options = dict(device=value.device, dtype=value.real.dtype)
    window = torch.hann_window(n, periodic=False, **options)
    density = torch.fft.fft(value*window, norm='ortho') * (WIDE/(n*window.square().mean())).sqrt()
    frequency = (n*torch.fft.fftfreq(n, **options)).abs()
    return density[..., (frequency >= 72) & (frequency <= 112)]


def guard_variance(y):
    return shared_guard_coefficients(y).abs().square().median(-1, keepdim=True).values/.69314718056


def gate_score(variance):
    # Both views support the same six central chirps.
    central = variance if variance.shape[1] == 6 else variance[:, 1:7]
    return central.sqrt().median(1).values.squeeze(-1)


class NarrowTracker(PacketRestorer):
    def __init__(self):
        super().__init__('consensus', samples=NARROW)
        index, taps = front_end(NARROW)
        self.register_buffer('indices', tensor(index[1:7]), persistent=False)
        self.register_buffer('taps', tensor(taps), persistent=False)

    def aperture(self, raw):
        return (raw[..., self.indices] * self.taps).sum(-1)


class ApertureTrackers(nn.Module):
    def __init__(self):
        super().__init__()
        self.narrow = NarrowTracker()
        self.wide = WideTracker()


class DualApertureDura(DuraCapacity):
    def __init__(self):
        super().__init__('expanded', condition_fields=8)

    def propose(self, y, quality, rms, edges, *, measurement=None):
        n = y.shape[-1]
        if measurement is None:
            measurement = guard_variance(y)
        # Convert the shared per-bin estimate back to the native per-sample scale.
        noise = measurement * (n/WIDE)
        if edges.shape[-1]:
            observed = boundary_fields(edges, rms)
            available = torch.ones_like(rms)
        else:
            available = torch.zeros_like(rms)
            observed = (available,) * 6
        resolution = torch.full_like(rms, n/WIDE)
        return self.stages(y, quality, rms, condition=(*observed, available, resolution),
                           noise_variance=noise)

    @staticmethod
    def frame(trackers, observed):
        if observed.shape[1:] == (6, 2, NARROW):
            return NARROW, coordinates(trackers.narrow, observed)
        if observed.shape[1:] == (8, 2, WIDE+2*EDGE):
            return WIDE, coordinates(trackers.wide, observed[..., :WIDE])
        raise ValueError('Expected recorded narrow IQ or W24 IQ with measured boundaries')

    def restore(self, trackers, observed, threshold):
        n, (y, phase, quality, rms) = self.frame(trackers, observed)
        variance = guard_variance(y)
        mask = gate_score(variance) > threshold
        if not mask.any():
            return observed, mask
        value = y[mask]
        edge = complex_iq(observed[mask, ..., n:])
        stages, *_ = self.propose(value, quality[mask], rms[mask], edge, measurement=variance[mask])
        output = observed.clone()
        output[mask, ..., :n] += real_iq(.5*(stages[-1]-value)*phase[mask]*rms[mask])
        return output, mask


@torch.no_grad()
def calibrate(model, trackers, raw):
    levels = []
    for batch in raw.split(32):
        for tracker, n in ((trackers.narrow, NARROW), (trackers.wide, WIDE)):
            wave = real_iq(tracker.aperture(batch)[..., :n])
            y, *_ = coordinates(tracker, wave)
            levels.append(gate_score(guard_variance(y)).max())
    return 1.05 * float(torch.stack(levels).max()), {'gate_calibration': '72–112 kHz pooled six-chirp RMS',
                                                     'parameters': sum(p.numel() for p in model.parameters())}


def objective(model, trackers, raw, rng, *, with_identity, recipe, step):
    first, second = raw.chunk(2)
    narrow, narrow_metrics = waveform_objective(model, trackers.narrow, first, rng,
                                                with_identity=with_identity, recipe=recipe, step=step,
                                                samples=NARROW, identity_rows=1)
    wide, wide_metrics = waveform_objective(model, trackers.wide, second, rng,
                                            with_identity=with_identity, recipe=recipe, step=step, identity_rows=1)
    loss = (len(first)*narrow + len(second)*wide)/len(raw)
    return loss, torch.cat((narrow_metrics, wide_metrics))


RECIPE = replace(W24_RECIPE, name='dual_aperture', model=DualApertureDura, finalize_calibration=calibrate)
