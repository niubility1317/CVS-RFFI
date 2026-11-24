"""Attribute LoRa phase and amplitude decisions on the already scored source panel."""
import json
from pathlib import Path

import numpy as np

from experiments.cric_dura_comparison.io import read
from experiments.pamf_lora_main.chirp_spectrum_source import read_frozen
from experiments.pamf_lora_main.data import SOURCE, VIEWS
from experiments.pamf_lora_main.full_packet_views import source_contract
from experiments.pamf_lora_main.independent_shape_source import MODES as CONTROL_MODES, RUN as CONTROL
from experiments.pamf_lora_main.polar_shape_source import MODES, RUN
from experiments.pamf_lora_main.spectral_unroll import ROUTES


def counts(actual, proposed, control):
    return dict(correct=int(proposed.sum()), repair=int((~actual & proposed).sum()),
                damage=int((actual & ~proposed).sum()),
                net_vs_direct=int(proposed.sum() - actual.sum()),
                net_vs_dura=int(proposed.sum() - control.sum()))


def main():
    runtime = read(Path('runtime.json'))
    current, source = (Path(runtime[key]) for key in ('run_root', 'source_root'))
    if current.name != RUN:
        raise ValueError('Unexpected source run')
    parent = current.parent / CONTROL
    if any(read(root / 'state.json')['status'] != 'SOURCE_SCORED' for root in (current, parent)):
        raise ValueError('All predictions and independent scores must be complete')
    candidate = {route: read_frozen(current, route, MODES) for route in ROUTES}
    reference = {route: read_frozen(parent, route, CONTROL_MODES) for route in ROUTES}
    ids = candidate['full'][0]
    if any(entry[0] != ids for entry in (*candidate.values(), *reference.values())):
        raise ValueError('Expected identical frozen source IDs')
    for route in ROUTES:
        for view in VIEWS:
            for name in MODES[:4]:
                found = candidate[route][1][view][MODES.index(name)]
                expected = reference[route][1][view][CONTROL_MODES.index(name)]
                if not np.array_equal(found, expected):
                    raise ValueError(f'Frozen control differs on {route}/{view}/{name}')
    lookup = {ident: i for i, ident in enumerate(source_contract(source)['role_ids']['V'])}
    selected = [lookup[ident] for ident in ids]
    with np.load(source / 'V_metadata.npz', allow_pickle=False) as labels:
        if labels['ids'][selected].tolist() != ids:
            raise ValueError('Source truth and fixed predictions disagree on physical IDs')
        truth, receiver = labels['y'][selected], labels['rx_i'][selected]
    rng = np.random.default_rng(39278)
    groups = [np.flatnonzero(receiver == i) for i in range(len(SOURCE))]
    indexes = np.concatenate([rng.choice(group, (1600, len(group)), replace=True) for group in groups], axis=1)
    results = {}
    for route in ROUTES:
        hits = np.stack([candidate[route][1][view] == truth for view in VIEWS])
        measures = {}
        for k, mode in enumerate(MODES):
            urban = hits[-1, k].astype(np.int32) - hits[-1, 1]
            six = hits[1:, k].sum(0) - hits[1:, 1].sum(0)
            difference_half = hits[1:, k].sum(0) - hits[1:, 3].sum(0)
            intervals = {}
            for label, values in [('six_vs_dura', six), ('urban_vs_dura', urban), ('six_vs_original_half', difference_half)]:
                intervals[label] = np.quantile(values[indexes].sum(1), [.025,.975]).tolist()
            measures[mode] = dict(
                six=counts(hits[1:, 0], hits[1:, k], hits[1:, 1]),
                by_view={name:counts(hits[j,0], hits[j,k], hits[j,1]) for j,name in enumerate(VIEWS)},
                by_receiver={rx:counts(hits[1:,0,g], hits[1:,k,g], hits[1:,1,g])
                             for rx,g in zip(SOURCE, groups, strict=True)},
                paired95=intervals,
            )
        results[route] = measures
    clean = read(current / 'clean-complete-scores.json')
    print(json.dumps(dict(status='VERIFIED', n=len(ids), target_reads=0, truth_after_predictions=True,
          frozen_controls_identical=True, results=results, clean=clean), allow_nan=False))


if __name__ == '__main__':
    main()
