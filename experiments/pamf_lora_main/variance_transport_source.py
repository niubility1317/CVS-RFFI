"""Score calibrated LoRa variance transport on the fixed dual-route source panel."""

from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, write

from .channel_blind import corruption_score
from .channel_blind_source import clean_threshold, panel, receive_only, setup
from .complex_vector_source import load_reader, score
from .data import SOURCE, VIEWS
from .dura_repair import PacketRestorer, complex_iq, real_iq
from .joint_unroll_source import coordinates, source_control
from .local_phase import generic_phase_observation
from .local_wiener_source import freeze, main
from .variance_transport import VarianceTransport


RUN = '20261124-phase1-lora-variance-transport-source-m2-r01'
MODES = ('direct', 'reference', 'posterior_mean', 'rms_matched_mean', 'variance_transport',
         'transport_shared_only', 'transport_private_only')
COMPARISONS = (('variance_transport', 'rms_matched_mean'), ('variance_transport', 'posterior_mean'),
               ('rms_matched_mean', 'posterior_mean'), ('variance_transport', 'direct'),
               ('variance_transport', 'transport_shared_only'), ('variance_transport', 'transport_private_only'),
               *((mode, 'reference') for mode in MODES[2:]))
FIELDS = ('active', 'common_outer_variance', 'private_outer_variance', *MODES[2:])
DEVICE = 'cuda:0'
CALIBRATION = dict(fit=5000, calibration=2500, held=500, optimizer_updates=0,
                   trainable_parameters=0, F_reads=0, V_reads=0, TX_reads=0, target_reads=0, exact_reads=0)


@torch.no_grad()
def calibrate(rt):
    started = perf_counter()
    model = VarianceTransport().to(DEVICE).eval()
    tracker = PacketRestorer('consensus').to(DEVICE).eval()
    (fit, calibration, held), contract = setup(rt, DEVICE, fit_per_rx=1000)
    framed = torch.cat([coordinates(tracker, rows)[0] for rows in fit.split(64)])
    model.calibrate(framed)
    threshold = float(clean_threshold(tracker, calibration))
    held_power = torch.cat([model.observation_noise(model.analysis(coordinates(tracker, x)[0]))
                            for x in held.split(64)])
    if (len(fit), len(calibration), len(held)) != (5000, 2500, 500):
        raise ValueError('The LoRa source U roles must match the frozen matrix')
    if not torch.isfinite(model.variance).all() or not np.isfinite(threshold) or threshold <= 0:
        raise ValueError('Invalid U-only covariance or threshold')
    folder = Path(rt['run_root'])
    torch.save(dict(contract=contract, weights=model.state_dict(), threshold=threshold), folder/'calibration.pt')
    write(folder/'calibration.json', dict(**CALIBRATION, threshold=threshold, wall_seconds=perf_counter()-started,
          held_outer_power_median=held_power.median(dim=0).values.tolist(),
          eigenvalue_median=model.variance.flatten(1).median(dim=1).values.tolist()))


def load_gate(root):
    data = read(root/'calibration.json')
    if any(data[key] != count for key, count in CALIBRATION.items()):
        raise ValueError('The source-only analytic calibration must match the preregistration')
    return data['threshold']


def load(root, contract, threshold, device=DEVICE):
    frozen = torch.load(root/'calibration.pt', map_location=device, weights_only=True)
    if frozen['contract'] != contract or frozen['threshold'] != threshold:
        raise ValueError('Analytic readout must use its own frozen source contract')
    model = VarianceTransport().to(device).eval()
    model.load_state_dict(frozen['weights'], strict=True)
    return model


def recover(iq, tracker, model, threshold):
    y, rotation, _, rms = coordinates(tracker, iq)
    keep = corruption_score(y) > threshold
    summaries = iq.new_zeros((len(iq), len(FIELDS)))
    output = iq.expand(len(MODES) - 2, *iq.shape).clone()
    count = int(keep.sum())
    if count:
        delta, noise, doses = model.changes(y[keep], rms[keep])
        physical = delta * (rotation[keep] * rms[keep])
        output[:, keep] = iq[keep] + torch.stack((physical.real, physical.imag), dim=-2)
        summaries[keep] = torch.cat((torch.ones_like(noise[:, :1]), noise[:, :2], doses), dim=1)
    return output, count, summaries


def render(iq, _unused, baseline, tracker, models, *, threshold, measurements=None):
    reference, _ = baseline(iq)
    output, count, extras = recover(iq, tracker, models[0], threshold)
    if measurements is not None:
        measurements.append(np.asarray(extras.tolist()))
    return torch.cat((torch.stack((iq, reference)), output)), count


@torch.no_grad()
def smoke(rt, threshold):
    root = Path(rt['run_root'])
    (_, sample, _), contract = setup(rt, DEVICE, smoke=True)
    _, _, expected, baseline, tracker = source_control(rt, 'full', DEVICE)
    if contract != expected:
        raise ValueError('Expected matching real source checkpoints')
    reader = load_reader(root, contract, DEVICE)
    model = load(root, contract, threshold)
    rng = torch.Generator(device=DEVICE).manual_seed(2026112404)
    unclean = real_iq(generic_phase_observation(complex_iq(sample), rng))
    disturbed = unclean[corruption_score(coordinates(tracker, unclean)[0]).argmax().view(1)]
    clean = sample[corruption_score(coordinates(tracker, sample)[0]).argmin().view(1)]
    incoming = torch.cat((clean, disturbed))
    waves, active = render(incoming, None, baseline, tracker, (model,), threshold=threshold)
    logits = reader(waves.flatten(0, 1))[0]
    if logits.shape != (2 * len(MODES), 10) or not torch.isfinite(logits).all() or active != 1:
        raise ValueError('No-query smoke requires a real frozen reader with both bypass and recovery')
    if not torch.equal(waves[2:, 0], clean.expand_as(waves[2:, 0])):
        raise ValueError('Clean bypass must be exact')
    times = {}
    for label, value in (('clean', clean), ('generic_disturbed', disturbed)):
        samples = []
        for index in range(28):
            torch.cuda.synchronize()
            start = perf_counter()
            recover(value, tracker, model, threshold)
            torch.cuda.synchronize()
            if index >= 8:
                samples.append((perf_counter() - start) * 1000)
        times[label] = float(np.median(samples))
    write(root/'smoke.json', dict(status='PASS', median_b1_ms=times,
          device=torch.cuda.get_device_name(), target_reads=0, V_reads=0))


def evaluate(rt, route, threshold):
    binding = source_control(rt, route, DEVICE)
    model = load(Path(rt['run_root']), binding[2], threshold)
    measurements = []

    def finish(root, route, source, chosen, ids, predictions, metrics, modes):
        values = np.concatenate(measurements).reshape(len(VIEWS), len(chosen), len(FIELDS))
        result = {}
        for view, rows in zip(VIEWS, values, strict=True):
            per_rx = {}
            for rx, group in zip(SOURCE, np.split(rows, len(SOURCE), axis=0), strict=True):
                active = group[group[:, 0] > 0.]
                summary = {'active': len(active)}
                if len(active):
                    summary.update(zip(FIELDS[1:], np.median(active[:, 1:], axis=0).tolist(), strict=True))
                per_rx[rx] = summary
            result[view] = per_rx
        write(root/f'{route}-estimates.json', result)
        freeze(root, route, source, chosen, ids, predictions, metrics, modes)

    score(rt, route, panel=panel, modes=MODES, binding=binding, frozen_models=(model,), replay_fn=receive_only,
          render=partial(render, threshold=threshold, measurements=measurements),
          require_clean_bypass=False, finalize=finish)


if __name__ == '__main__':
    main(run=RUN, runner=__spec__.name, modes=MODES, comparisons=COMPARISONS, fit=calibrate,
         load_gate=load_gate, check=smoke, evaluate=evaluate)
