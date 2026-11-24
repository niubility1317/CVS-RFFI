"""Score an analytic cross-frequency prior with the frozen LoRa Tri-set reader."""

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
from .data import SOURCE, VIEWS
from .dura_repair import PacketRestorer, complex_iq, real_iq
from .joint_unroll_source import coordinates, source_control
from .local_wiener_source import freeze, main
from .phase_dose_bursts_source import burst_observation
from .variance_transport import physical_norm


RUN = '20261124-phase1-lora-cross-frequency-source-m2-r01'
DEVICE = 'cuda:0'
MODES = ('direct', 'reference', 'joint_mean', 'bin_mean', 'joint_transport', 'bin_transport',
         'joint_equal', 'bin_equal')
COMPARISONS = (('joint_equal', 'bin_equal'), ('joint_transport', 'bin_transport'),
               ('joint_mean', 'bin_mean'), ('joint_transport', 'joint_mean'),
               *((mode, control) for mode in MODES[2:] for control in ('direct', 'reference')))
FIELDS = ('active', 'common_quadrature_variance', 'private_quadrature_variance', *MODES[2:])
CALIBRATION = dict(fit=20000, calibration=2500, held=500, optimizer_updates=0,
                   trainable_parameters=0, F_reads=0, V_reads=0, TX_reads=0, target_reads=0, exact_reads=0)


@torch.no_grad()
def calibrate(rt):
    started = perf_counter()
    model = CrossFrequencyPrior().to(DEVICE).eval()
    tracker = PacketRestorer('consensus').to(DEVICE).eval()
    (fit, calibration, held), contract = setup(rt, DEVICE, fit_per_rx=4000)
    if (len(fit), len(calibration), len(held)) != (20000, 2500, 500):
        raise ValueError('Expected the preregistered source U role sizes')
    model.calibrate(coordinates(tracker, batch)[0] for batch in fit.split(96))
    threshold = float(clean_threshold(tracker, calibration))
    held_power = torch.cat([model.noise(model.spectrum(coordinates(tracker, batch)[0])) for batch in held.split(64)])
    state = model.state_dict()
    if not all(torch.isfinite(x).all() for x in state.values()) or not np.isfinite(threshold):
        raise ValueError('Nonfinite source-only covariance or gate')
    folder = Path(rt['run_root'])
    torch.save(dict(contract=contract, threshold=threshold, weights=state), folder/'calibration.pt')
    write(folder/'calibration.json', dict(**CALIBRATION, threshold=threshold, wall_seconds=perf_counter()-started,
          prior_floats=sum(value.numel() for value in state.values()),
          held_noise_median=held_power.median(0).values.tolist(),
          joint_top16_fraction=(model.joint_variance[:, -16:].sum(1) / model.joint_variance.sum(1)).tolist()))


def load_gate(root):
    data = read(root/'calibration.json')
    if any(data[field] != value for field, value in CALIBRATION.items()):
        raise ValueError('Expected the registered source-only calibration')
    return data['threshold']


def load(root, contract, threshold, device=DEVICE):
    saved = torch.load(root/'calibration.pt', map_location=device, weights_only=True)
    if saved['contract'] != contract or saved['threshold'] != threshold:
        raise ValueError('Calibration must be bound to the same source U contract')
    model = CrossFrequencyPrior().to(device).eval()
    model.load_state_dict(saved['weights'], strict=True)
    return model


def recover(iq, tracker, model, threshold):
    y, rotation, _, rms = coordinates(tracker, iq)
    active = corruption_score(y) > threshold
    count = int(active.sum())
    measurements = iq.new_zeros((len(iq), len(FIELDS)))
    output = iq.expand(len(MODES) - 2, *iq.shape).clone()
    if count:
        wave, scale = y[active], rms[active]
        change, noise = model.changes(wave, scale)
        physical = change * (rotation[active] * scale)
        output[:, active] = iq[active] + torch.stack((physical.real, physical.imag), dim=-2)
        dose = physical_norm(change, scale) / physical_norm(wave, scale).clamp_min(1e-10)
        measurements[active] = torch.cat((torch.ones_like(noise[:, :1]), noise, dose[..., 0, 0].mT), 1)
    return output, count, measurements


def render(iq, _unused, baseline, tracker, models, *, threshold, measurements=None):
    reference, _ = baseline(iq)
    output, count, info = recover(iq, tracker, models[0], threshold)
    if measurements is not None:
        measurements.append(np.asarray(info.tolist()))
    return torch.cat((torch.stack((iq, reference)), output)), count


@torch.no_grad()
def smoke(rt, threshold, *, modes=MODES, loader=load, renderer=render, recovery=recover):
    root = Path(rt['run_root'])
    (_, inputs, _), contract = setup(rt, DEVICE, smoke=True)
    _, _, expected, baseline, tracker = source_control(rt, 'full', DEVICE)
    if expected != contract:
        raise ValueError('Expected matching frozen LoRa checkpoints')
    reader = load_reader(root, contract, DEVICE)
    model = loader(root, contract, threshold)
    rng = torch.Generator(device=DEVICE).manual_seed(2026112477)
    disturbed = real_iq(burst_observation(complex_iq(inputs), rng))
    noisy = disturbed[corruption_score(coordinates(tracker, disturbed)[0]).argmax().view(1)]
    clean = inputs[corruption_score(coordinates(tracker, inputs)[0]).argmin().view(1)]
    output, active = renderer(torch.cat((clean, noisy)), None, baseline, tracker, (model,), threshold=threshold)
    logits = reader(output.flatten(0, 1))[0]
    if (logits.shape != (2 * len(modes), 10) or not torch.isfinite(logits).all() or active != 1
            or not torch.equal(output[2:, 0], clean.expand_as(output[2:, 0]))):
        raise ValueError('Expected real frozen reader logits, an exact bypass and one active recovery')
    times = {}
    for label, sample in (('clean', clean), ('generic_disturbed', noisy)):
        elapsed = []
        for step in range(28):
            torch.cuda.synchronize()
            start = perf_counter()
            recovery(sample, tracker, model, threshold)
            torch.cuda.synchronize()
            if step >= 8:
                elapsed.append(1000 * (perf_counter() - start))
        times[label] = float(np.median(elapsed))
    write(root/'smoke.json', dict(status='PASS', median_b1_ms=times,
          device=torch.cuda.get_device_name(), V_reads=0, target_reads=0))


def evaluate(rt, route, threshold, *, modes=MODES, fields=FIELDS, loader=load, renderer=render):
    binding = source_control(rt, route, DEVICE)
    model = loader(Path(rt['run_root']), binding[2], threshold)
    measurements = []

    def finish(root, route, source, chosen, ids, predictions, metrics, modes):
        grouped = np.concatenate(measurements).reshape(len(VIEWS), len(SOURCE), -1, len(fields))
        summaries = {}
        for view, rows in zip(VIEWS, grouped, strict=True):
            result = {}
            for rx, row in zip(SOURCE, rows, strict=True):
                active = row[row[:, 0] > 0]
                stats = dict(zip(fields[1:], np.median(active[:, 1:], axis=0).tolist(), strict=True)) if len(active) else {}
                result[rx] = dict(active=len(active), **stats)
            summaries[view] = result
        write(root/f'{route}-estimates.json', dict(by_view=summaries))
        freeze(root, route, source, chosen, ids, predictions, metrics, modes)

    score(rt, route, panel=panel, modes=modes, binding=binding, frozen_models=(model,), replay_fn=receive_only,
          render=partial(renderer, threshold=threshold, measurements=measurements),
          require_clean_bypass=False, finalize=finish)


if __name__ == '__main__':
    main(run=RUN, runner=__spec__.name, modes=MODES, comparisons=COMPARISONS, fit=calibrate,
         load_gate=load_gate, check=smoke, evaluate=evaluate)
