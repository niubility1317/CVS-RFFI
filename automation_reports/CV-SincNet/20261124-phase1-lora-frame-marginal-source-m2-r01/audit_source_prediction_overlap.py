"""Attribute same-packet errors using only completed LoRa source predictions."""
import json
from pathlib import Path

import numpy as np

from experiments.cric_dura_comparison.io import read
from experiments.pamf_lora_main.channel_blind_source import panel
from experiments.pamf_lora_main.chirp_spectrum_source import read_frozen, source_contract
from experiments.pamf_lora_main.data import SOURCE, VIEWS
from experiments.pamf_lora_main.frame_marginal_source import MODES


ENDPOINTS = ('direct', 'reference', 'rank_half', 'bounded_half')
INDEX = [MODES.index(name) for name in ENDPOINTS]
ROUTES = ('full', 'residual')


def summarize(four, marginal, clean_correct):
    any_correct = four.any(axis=0)
    wrong = ~any_correct
    correct_count = four.sum(axis=0)
    return dict(
        n=int(wrong.size),
        correct={name: int(c.sum()) for name, c in zip(ENDPOINTS, four, strict=True)},
        all_correct=int((correct_count == len(ENDPOINTS)).sum()),
        any_correct=int(any_correct.sum()),
        only_this_correct={name: int((c & (correct_count == 1)).sum())
                           for name, c in zip(ENDPOINTS, four, strict=True)},
        uniquely_added_by_frame=int((marginal & wrong).sum()),
        all_wrong=int(wrong.sum()),
        all_wrong_clean_correct=int((wrong & clean_correct).sum()),
        clean_correct=int(clean_correct.sum()))


def audit(runtime):
    root = Path(runtime['run_root'])
    source = Path(runtime['source_root'])
    if read(root / 'state.json')['status'] != 'SOURCE_SCORED':
        raise ValueError('Expected a completed source run')
    frozen = {route: read_frozen(root, route, MODES) for route in ROUTES}
    ids = frozen['full'][0]
    if ids != frozen['residual'][0]:
        raise ValueError('Expected the same physical panel in both routes')
    contract_ids = source_contract(source)['role_ids']['V']
    positions = {identity: i for i, identity in enumerate(contract_ids)}
    selected = np.asarray([positions[identity] for identity in ids])
    with np.load(source / 'V_metadata.npz', allow_pickle=False) as truth:
        rx = np.asarray(SOURCE)[truth['rx_i']]
        if not np.array_equal(selected, panel(rx)):
            raise ValueError('Unexpected source V panel')
        labels = truth['y'][selected]

    views = VIEWS[1:]
    correct = {route: np.stack([frozen[route][1][v] == labels for v in views], axis=1)
               for route in ROUTES}
    clean = frozen['full'][1]['clean'][MODES.index('direct')] == labels
    if not np.array_equal(frozen['full'][1]['clean'][0], frozen['residual'][1]['clean'][0]):
        raise ValueError('Direct clean predictions disagree across routes')

    results = {}
    for route, matrix in correct.items():
        frame = matrix[MODES.index('frame_marginal')]
        results[route] = dict(six=summarize(matrix[INDEX], frame, np.tile(clean, (len(views), 1))),
                              by_view={v: summarize(matrix[INDEX, i], frame[i], clean)
                                       for i, v in enumerate(views)})
    common = np.logical_and.reduce([~correct[r][INDEX].any(axis=0) for r in ROUTES])
    common_clean = common & clean
    return dict(status='SOURCE_POSTHOC_COMPLETE', source_packets=len(ids), routes=results,
                common_wrong=dict(six=int(common.sum()), low_urban=int(common[-1].sum()),
                                  six_clean_correct=int(common_clean.sum()), low_urban_clean_correct=int(common_clean[-1].sum())),
                uses_truth_for_attribution_only=True, target_reads=0,
                new_predictions=0, training_updates=0)


if __name__ == '__main__':
    print(json.dumps(audit(read(Path('runtime.json'))), allow_nan=False))
