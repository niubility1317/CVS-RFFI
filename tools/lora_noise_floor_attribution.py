"""Explain the registered LoRa Wiener floor bias inside an isolated source audit."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
from math import log
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor
from experiments.pamf_lora_main.channel_blind_source import panel, receive_only
from experiments.pamf_lora_main.data import SOURCE, VIEWS
from experiments.pamf_lora_main.denoise_source import replay
from experiments.pamf_lora_main.dura_repair import PacketRestorer, complex_iq
from experiments.pamf_lora_main.full_packet_publish import DATA_RUN
from experiments.pamf_lora_main.full_packet_views import native, source_contract
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.local_wiener import LocalWiener
from experiments.pamf_lora_main.noise_floor_diagnostic import RUN
from experiments.pamf_lora_main.runner import activate


FIELDS = ('floor_over_true_total', 'observed_over_noise_floor', 'pooled_local_variance',
          'standardized_median', 'repeat_demeaning', 'true_outer_over_total')


@torch.no_grad()
def ratios(model, tracker, wave, added):
    """Return positive multiplicative factors whose product is the original bias."""
    observed, phase, _, rms = coordinates(tracker, wave)
    _, floor, _ = model.analyze(observed)
    known, _, _ = model.analyze(complex_iq(added)/(phase*rms))
    outer = known.flatten(-2)[..., model.outer]
    repeats = outer.shape[1]
    centered = (outer-outer.mean(1, keepdim=True)) / (1.-1./repeats)**.5
    power = centered.abs().square()
    local = power.mean(-1, keepdim=True)
    standardized = (power/local.clamp_min(1e-12)).flatten(1).median(1).values / log(2.)
    noise_floor = power.flatten(1).median(1).values / log(2.)
    centered_power = power.mean((1, 2, 3))
    edge_power = outer.abs().square().mean((1, 2, 3))
    total = known.abs().square().mean((1, 2, 3, 4))
    factors = (floor/total, floor/noise_floor, noise_floor/(centered_power*standardized),
               standardized, centered_power/edge_power, edge_power/total)
    report = torch.stack(factors, -1)
    if not torch.isfinite(report).all() or (report <= 0).any():
        raise FloatingPointError('Nonpositive source noise attribution')
    if not torch.allclose(report[:, 0], report[:, 1:].prod(-1), atol=1e-6, rtol=1e-6):
        raise ValueError('Noise floor factors no longer close per packet')
    return report


def summarize(rows):
    packs = {}
    for key, column in zip(FIELDS, rows.T, strict=True):
        packs[key] = dict(zip(('p10_db', 'p50_db', 'p90_db'),
                              np.quantile(10.*np.log10(column), (.1, .5, .9)).tolist(), strict=True),
                          mean_db=float(np.mean(10.*np.log10(column))),
                          median_ratio=float(np.median(column)))
    closure = packs[FIELDS[0]]['mean_db']-sum(packs[key]['mean_db'] for key in FIELDS[1:])
    if abs(closure) > 1e-5:
        raise ValueError('Mean dB attribution does not close')
    return dict(rows=len(rows), closure_db=closure, values=packs)


def audit(rt):
    started = perf_counter()
    root = Path(rt['run_root'])
    source = Path(rt['source_root'])
    if rt['run_id'] != RUN or read(root/'state.json')['status'] != 'SOURCE_SCORED':
        raise ValueError('Expected the completed isolated noise floor experiment')
    activate(rt)
    contract = source_contract(source)
    count = contract['counts']['V']
    raw = native(source, 'V', count)
    with np.load(source/'V_metadata.npz', allow_pickle=False) as metadata:
        ids = metadata['ids']
        rx = np.asarray(SOURCE)[metadata['rx_i']]
    if ids.tolist() != contract['role_ids']['V']:
        raise ValueError('Source V physical ID order differs')
    chosen = panel(rx)
    chosen_rx = rx[chosen]
    packets = list(zip(raw[chosen], chosen_rx, ids[chosen], strict=True))
    cache = np.load(root.parent/DATA_RUN/'source_views'/'V.npy', mmap_mode='r', allow_pickle=False)
    if cache.shape != (len(VIEWS), count, 6, 2, 256) or cache.dtype != np.float32:
        raise ValueError('Expected the registered Full practical cache')
    wiener, tracker = LocalWiener().eval(), PacketRestorer('consensus').eval()
    result = {}
    with ProcessPoolExecutor(max_workers=4) as pool:
        for route in ('full', 'residual'):
            result[route], combined = {}, []
            original = read(root/f'{route}-isolated-estimates.json')
            for index, view in enumerate(VIEWS[1:], 1):
                capture, noise = (np.stack(x) for x in zip(*pool.map(partial(replay, route, view), packets,
                                                                   chunksize=12), strict=True))
                if route == 'full':
                    same_iq = np.array_equal(capture, cache[index, chosen])
                else:
                    first_per_rx = range(0, len(packets), len(packets)//len(SOURCE))
                    same_iq = all(np.array_equal(capture[i], receive_only(route, view, packets[i])[0])
                                  for i in first_per_rx)
                if not same_iq:
                    raise ValueError('Noise audit altered the registered receive IQ')
                values = [ratios(wiener, tracker, a, b).tolist() for a, b in
                          zip(tensor(capture).split(32), tensor(noise).split(32), strict=True)]
                values = np.asarray([row for chunk in values for row in chunk])
                report = summarize(values)
                previous = original['estimates'][view]
                for key, prior in (('floor_over_true_total', 'estimate_ratio'),
                                   ('true_outer_over_total', 'true_outer_to_total')):
                    if abs(report['values'][key]['median_ratio']-previous[prior][1]) > 2e-4:
                        raise ValueError('CPU audit does not reproduce the archived GPU noise statistic')
                result[route][view] = dict(all=report,
                                           by_receiver={key: summarize(values[chosen_rx == key]) for key in SOURCE})
                combined.append(values)
            result[route]['six'] = summarize(np.concatenate(combined))
    return dict(status='VERIFIED', diagnostic_only=True, run=RUN, basis_commit=rt['code_commit'],
                source_V_rows=len(chosen), routes=result, seconds=perf_counter()-started,
                F_reads=0, TX_reads=0, target_reads=0, optimizer_updates=0, calibration_updates=0,
                export='aggregate diagnostic statistics only')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runtime', type=Path)
    print(json.dumps(audit(read(parser.parse_args().runtime)), allow_nan=False))
