"""Audit held-out relative-action evidence on the completed LoRa source V panel."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor
from experiments.pamf_lora_main.channel_blind_source import panel, receive_only
from experiments.pamf_lora_main.data import SOURCE, VIEWS
from experiments.pamf_lora_main.dura_repair import PacketRestorer
from experiments.pamf_lora_main.full_packet_publish import DATA_RUN
from experiments.pamf_lora_main.full_packet_views import native, source_contract
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.receive_frame_action import estimate, opposite_fold
from experiments.pamf_lora_main.receive_frame_action_source import RUN, gate_value
from experiments.pamf_lora_main.runner import activate


@torch.no_grad()
def measure(iq, tracker):
    rows = []
    for wave in tensor(iq, 'cpu').split(32):
        y, phase, *_ = coordinates(tracker, wave)
        reference, folds, unit = opposite_fold(y)
        _, tracked = estimate(y, phase, reference, y, folds, relative=True, raw_frame=False)
        _, physical = estimate(y, phase, reference, y, folds, relative=True, raw_frame=True)
        rows.extend(torch.stack((unit, tracked, physical), -1).tolist())
    return np.asarray(rows)


def quantiles(x):
    names = ('p05', 'p50', 'p90', 'p95', 'p99', 'max')
    return dict(zip(names, np.quantile(x, (.05, .5, .9, .95, .99, 1.)).tolist(), strict=True))


def describe(rows, anchors):
    unit, tracked, physical = rows.reshape(-1, 3).T
    relative = (unit-physical)/np.maximum(unit, 1e-9)
    return dict(rows=len(unit), baseline_error=quantiles(unit),
                raw_signed_fraction=quantiles(relative),
                raw_absolute_reduction=quantiles(unit-physical),
                raw_minus_tracked_fraction=quantiles((tracked-physical)/np.maximum(unit, 1e-9)),
                raw_better_than_tracked=int(np.count_nonzero(physical < tracked)),
                raw_above={name: int(np.count_nonzero(relative > cutoff)) for name, cutoff in anchors.items()})


def source_panel(rt):
    run = Path(rt['run_root'])
    source = Path(rt['source_root'])
    if rt['run_id'] != RUN or read(run/'state.json')['status'] != 'SOURCE_SCORED':
        raise ValueError('Expected the finished receive-frame source experiment')
    activate(rt)
    contract = source_contract(source)
    count = contract['counts']['V']
    raw = native(source, 'V', count)
    with np.load(source/'V_metadata.npz', allow_pickle=False) as fields:
        ids = fields['ids']
        rx = np.asarray(SOURCE)[fields['rx_i']]
    if ids.tolist() != contract['role_ids']['V']:
        raise ValueError('Source physical ID order differs')
    chosen = panel(rx)
    selected_rx = rx[chosen]
    captures = list(zip(raw[chosen], selected_rx, ids[chosen], strict=True))
    cached = np.load(run.parent/DATA_RUN/'source_views'/'V.npy', mmap_mode='r', allow_pickle=False)
    if cached.shape != (len(VIEWS), count, 6, 2, 256) or cached.dtype != np.float32:
        raise ValueError('Expected all registered Full practical source views')
    return ids[chosen], selected_rx, captures, cached, chosen


def view_pairs(captures, cached, chosen):
    with ProcessPoolExecutor(max_workers=4) as pool:
        for index, view in enumerate(VIEWS):
            full = cached[index, chosen]
            residual = np.stack([x for x, _ in pool.map(partial(receive_only, 'residual', view),
                                                        captures, chunksize=12)])
            if view == 'clean' and not np.array_equal(full, residual):
                raise ValueError('Both source routes must share the exact clean IQ')
            yield view, full, residual


def audit(rt):
    start = perf_counter()
    run = Path(rt['run_root'])
    ids, selected_rx, captures, cached, chosen = source_panel(rt)
    threshold = gate_value(run)
    tracker = PacketRestorer('consensus').eval()
    values = {route: {} for route in ('full', 'residual')}
    for view, full, residual in view_pairs(captures, cached, chosen):
        values['full'][view] = measure(full, tracker)
        values['residual'][view] = values['full'][view] if view == 'clean' else measure(residual, tracker)
    if any(not np.isfinite(x).all() for data in values.values() for x in data.values()):
        raise FloatingPointError('Nonfinite source held-out evidence')
    clean = values['full']['clean']
    clean_score = (clean[:, 0]-clean[:, 2])/np.maximum(clean[:, 0], 1e-9)
    anchors = {'positive': 0., '0.05': .05, '0.10': .10, '0.25': .25,
               'source_V_clean_p95': float(np.quantile(clean_score, .95)),
               'source_U_narrow_peak': read(run/'gate.json')['peaks'][0], 'frozen': threshold}
    result = {}
    for route, views in values.items():
        six = np.stack([views[view] for view in VIEWS[1:]])
        result[route] = dict(six=describe(six, anchors),
                             views={view: describe(rows, anchors) for view, rows in views.items()},
                             receiver_six={key: describe(six[:, selected_rx == key], anchors) for key in SOURCE})
        if any(row['raw_above']['frozen'] for row in result[route]['views'].values()):
            raise ValueError('Audit disagrees with the recorded zero-activation experiment')
    return dict(status='VERIFIED', diagnostic='SOURCE_V_ONLY', run=RUN, basis_commit=rt['code_commit'],
                ids=ids.tolist(), anchors=anchors, routes=result, seconds=perf_counter()-start,
                F_reads=0, TX_reads=0, target_reads=0, trace_reads=0, optimizer_updates=0, gate_changes=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runtime', type=Path)
    print(json.dumps(audit(read(parser.parse_args().runtime)), allow_nan=False))
