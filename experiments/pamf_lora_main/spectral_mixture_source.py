"""Fit the LoRa mixture on unlabeled source IQ and score both frozen routes."""

from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, write

from .channel_blind import corruption_score
from .channel_blind_source import clean_threshold, setup
from .cross_frequency_source import DEVICE, evaluate, smoke
from .data import SOURCE
from .dura_repair import PacketRestorer, complex_iq, real_iq
from .joint_unroll_source import coordinates
from .local_wiener_source import main
from .phase_dose_bursts_source import burst_observation
from .spectral_mixture import COMPONENTS, EM_STEPS, ROUTING_DIMENSIONS, SpectralMixture
from .variance_transport import physical_norm


RUN = '20261124-phase1-lora-spectral-mixture-source-m2-r01'
MODES = ('direct', 'reference', 'pooled_full', 'pooled_half', 'mixture_full', 'mixture_half',
         'split_half', 'guarded_half', 'split_equal', 'pooled_equal', 'prior_prob_half')
COMPARISONS = (('split_equal', 'pooled_equal'), ('guarded_half', 'pooled_half'),
               ('split_half', 'prior_prob_half'), ('split_half', 'mixture_half'),
               ('mixture_full', 'pooled_full'), ('guarded_half', 'split_half'),
               *((mode, baseline) for mode in MODES[2:] for baseline in ('direct', 'reference')))
ROUTING_FIELDS = ('max_posterior', 'half_probability_tv', 'posterior_vs_prior_tv', 'cross_trust')
FIELDS = ('active', 'common_quadrature_variance', 'private_quadrature_variance',
          *ROUTING_FIELDS, *MODES[2:])
PROOF = dict(fit=20000, calibration=2500, held=500, em_updates=EM_STEPS,
             components=COMPONENTS, dimensions=ROUTING_DIMENSIONS,
             F_reads=0, V_reads=0, TX_reads=0, target_reads=0, exact_reads=0)


@torch.no_grad()
def calibrate(rt):
    started = perf_counter()
    model = SpectralMixture().to(DEVICE).eval()
    tracker = PacketRestorer('consensus').to(DEVICE).eval()
    (fit, calibration, held), contract = setup(rt, DEVICE, fit_per_rx=4000)
    if tuple(map(len, (fit, calibration, held))) != (20000, 2500, 500):
        raise ValueError('Expected the preregistered U fit, calibration and held roles')

    def frames():
        return (coordinates(tracker, batch)[0] for batch in fit.split(96))

    result = model.fit(frames)
    threshold = float(clean_threshold(tracker, calibration))
    rng = torch.Generator(device=DEVICE).manual_seed(2026112477)
    summary = {}
    for rx, rows in zip(SOURCE, held.chunk(len(SOURCE)), strict=True):
        data = []
        for group in rows.split(32):
            received = real_iq(burst_observation(complex_iq(group), rng))
            tracked, _, _, _ = coordinates(tracker, received)
            _, info = model.routing(tracked)
            data.append(info)
        summary[rx] = dict(zip(ROUTING_FIELDS, torch.cat(data).median(0).values.tolist(), strict=True))
    state = model.state_dict()
    if not all(torch.isfinite(x).all() for x in state.values()) or not np.isfinite(threshold):
        raise ValueError('Nonfinite U-only mixture or bypass')
    root = Path(rt['run_root'])
    torch.save(dict(contract=contract, threshold=threshold, weights=state), root/'calibration.pt')
    write(root/'calibration.json', dict(**PROOF, **result, threshold=threshold,
          wall_seconds=perf_counter()-started, persistent_values=sum(x.numel() for x in state.values()),
          held_by_receiver=summary))


def load_gate(root):
    data = read(root/'calibration.json')
    if any(data[field] != value for field, value in PROOF.items()):
        raise ValueError('Expected the fixed source U mixture configuration')
    return data['threshold']


def load(root, contract, threshold, device=DEVICE):
    saved = torch.load(root/'calibration.pt', map_location=device, weights_only=True)
    if saved['contract'] != contract or saved['threshold'] != threshold:
        raise ValueError('The frozen mixture requires its source U contract and threshold')
    model = SpectralMixture().to(device).eval()
    model.load_state_dict(saved['weights'], strict=True)
    return model


def recover(iq, tracker, model, threshold):
    y, phase, _, rms = coordinates(tracker, iq)
    active = corruption_score(y) > threshold
    count = int(active.sum())
    out = iq.expand(len(MODES) - 2, *iq.shape).clone()
    values = iq.new_zeros((len(iq), len(FIELDS)))
    if count:
        frame, scale = y[active], rms[active]
        edit, noise, routing = model.changes(frame, scale)
        physical = edit * (phase[active] * scale)
        out[:, active] = iq[active] + torch.stack((physical.real, physical.imag), dim=-2)
        dose = physical_norm(edit, scale) / physical_norm(frame, scale).clamp_min(1e-9)
        values[active] = torch.cat((torch.ones_like(noise[:, :1]), noise, routing, dose[..., 0, 0].mT), 1)
    return out, count, values


def render(iq, _unused, baseline, tracker, models, *, threshold, measurements=None):
    reference, _ = baseline(iq)
    output, count, evidence = recover(iq, tracker, models[0], threshold)
    if measurements is not None:
        measurements.append(np.asarray(evidence.tolist()))
    return torch.cat((torch.stack((iq, reference)), output)), count


if __name__ == '__main__':
    main(run=RUN, runner=__spec__.name, modes=MODES, comparisons=COMPARISONS, fit=calibrate,
         load_gate=load_gate, check=partial(smoke, modes=MODES, loader=load, renderer=render, recovery=recover),
         evaluate=partial(evaluate, modes=MODES, fields=FIELDS, loader=load, renderer=render))
