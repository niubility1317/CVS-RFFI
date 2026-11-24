"""Freeze and score a receive-only LoRa short-echo engineering comparison."""

from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, write
from .channel_blind import corruption_score
from .channel_blind_source import clean_threshold, panel, receive_only, setup
from .complex_vector_source import load_reader, score
from .cross_frequency import CrossFrequencyPrior
from .data import VIEWS
from .dura_repair import PacketRestorer
from .joint_unroll_source import coordinates, source_control
from .local_wiener_source import freeze, main
from .relative_echo import correction, relative_models


RUN = '20261124-phase1-lora-relative-echo-source-m2-r01'
DEVICE = 'cuda:0'
MODES = ('direct', 'reference', 'scalar', 'short', 'echo_gate', 'echo_private')
COMPARISONS = (('short', 'scalar'), ('echo_gate', 'short'), ('echo_private', 'echo_gate'),
               *((mode, base) for mode in MODES[2:] for base in ('direct', 'reference')))


def observe(iq, tracker, prior):
    y, phase, _, rms = coordinates(tracker, iq)
    noise = 2. * prior.noise(prior.spectrum(y)).mean(-1).reshape(-1, 1, 1)
    return y, phase, rms, relative_models(y, phase, noise)


@torch.inference_mode()
def calibrate(rt):
    (_, calibration, _), _ = setup(rt, DEVICE, fit_per_rx=4000)
    tracker = PacketRestorer('consensus').to(DEVICE).eval()
    prior = CrossFrequencyPrior().to(DEVICE).eval()
    noise = float(clean_threshold(tracker, calibration))
    margin = 1.05 * max(float(observe(iq, tracker, prior)[3][2][4].max()) for iq in calibration.split(48))
    if not np.isfinite([noise, margin]).all() or noise <= 0 or margin <= 0:
        raise ValueError('Nonfinite or empty LoRa source U calibration')
    write(Path(rt['run_root'])/'gate.json',
          dict(status='SOURCE_U_FROZEN', n=len(calibration), noise=noise, echo=margin, target_reads=0))


def load_gate(root):
    gate = read(root/'gate.json')
    if gate['status'] != 'SOURCE_U_FROZEN' or gate['n'] != 2500:
        raise ValueError('Expected frozen LoRa U-only gate calibration')
    return gate['noise'], gate['echo']


def restore(iq, tracker, prior, gate):
    y, phase, rms, (scalar, short, evidence) = observe(iq, tracker, prior)
    noise_active = corruption_score(y) > gate[0]
    echo_active = noise_active | (evidence[4].flatten() > gate[1])
    switches = torch.stack((noise_active, noise_active, echo_active, echo_active))
    first, second = (correction(y, phase, action) for action in (scalar, short))
    private = second - second.mean(1, keepdim=True)
    physical = torch.stack((first, second, second, private)) * phase * rms
    output = iq + torch.stack((physical.real, physical.imag), dim=-2)
    output = torch.where(switches[..., None, None, None], output, iq)
    return output, switches


def render(iq, _unused, baseline, tracker, models, *, gate, triggered=None):
    reference, _ = baseline(iq)
    outputs, switches = restore(iq, tracker, models[0], gate)
    if triggered is not None:
        triggered.append(np.asarray(switches.tolist(), dtype=np.bool_))
    return torch.cat((torch.stack((iq, reference)), outputs)), int(switches[-1].sum())


@torch.no_grad()
def smoke(rt, gate):
    (_, calibration, _), contract = setup(rt, DEVICE, smoke=True)
    _, _, expected, baseline, tracker = source_control(rt, 'full', DEVICE)
    if expected != contract:
        raise ValueError('LoRa engineering smoke needs the matched source contract')
    root = Path(rt['run_root'])
    prior = CrossFrequencyPrior().to(DEVICE).eval()
    quiet = calibration[:1]
    rng = torch.Generator(device=DEVICE).manual_seed(2026112437)
    changed = quiet + .2*torch.randn(quiet.shape, device=DEVICE, generator=rng)
    waves, _ = render(torch.cat((quiet, changed)), None, baseline, tracker, (prior,), gate=gate)
    logits = load_reader(root, contract, DEVICE)(waves.flatten(0, 1))[0]
    if logits.shape != (2*len(MODES), 10) or not torch.isfinite(logits).all():
        raise ValueError('Invalid frozen-reader LoRa engineering smoke')
    time = {}
    for name, sample in (('clean', quiet), ('disturbed', changed)):
        measurements = []
        for iteration in range(28):
            torch.cuda.synchronize()
            start = perf_counter()
            restore(sample, tracker, prior, gate)
            torch.cuda.synchronize()
            if iteration >= 8:
                measurements.append(1000*(perf_counter()-start))
        time[name] = float(np.median(measurements))
    write(root/'smoke.json', dict(status='PASS', V_reads=0, target_reads=0, timing_b1_ms=time,
                                  trainable_parameters=0, device=torch.cuda.get_device_name()))


def evaluate(rt, route, gate):
    binding = source_control(rt, route, DEVICE)
    prior = CrossFrequencyPrior().to(DEVICE).eval()
    activity = []

    def finalize(root, route, source, chosen, ids, predictions, metrics, modes):
        data = np.concatenate(activity, axis=1).reshape(len(MODES)-2, len(VIEWS), len(chosen))
        counts = {view: dict(zip(MODES[2:], data[:, i].sum(1).tolist(), strict=True))
                  for i, view in enumerate(VIEWS)}
        write(root/f'{route}-activity.json', dict(by_view=counts))
        freeze(root, route, source, chosen, ids, predictions, metrics, modes)

    score(rt, route, panel=panel, modes=MODES, binding=binding, frozen_models=(prior,), replay_fn=receive_only,
          render=partial(render, gate=gate, triggered=activity), require_clean_bypass=False, finalize=finalize)


if __name__ == '__main__':
    main(run=RUN, runner=__spec__.name, modes=MODES, comparisons=COMPARISONS,
         fit=calibrate, load_gate=load_gate, check=smoke, evaluate=evaluate)
