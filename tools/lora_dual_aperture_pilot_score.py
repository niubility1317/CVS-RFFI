"""Independently score the two-route, single-seed dual-aperture LoRa pilot."""

import argparse
from pathlib import Path
import zipfile

import numpy as np

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main import dura_capacity_target as expanded, dura_dual_aperture_evaluate as pilot
from experiments.pamf_lora_main import residual_dura as residual, weak_guard_target as guard
from experiments.pamf_lora_main.data import TARGET, VIEWS, label_ids, packet_indices, recording
from experiments.pamf_lora_main.full_packet_data import LIMIT
from tools import lora_weak_guard_target_score as scoring


SOURCE_FILES = ('full-predictions.json', 'residual-predictions.json', 'full-source-scores.json',
                'residual-source-scores.json', 'source-promotion.json')


def collect(receipt, config, output):
    scoring.collect(receipt, config, output, replay=pilot, artifacts=SOURCE_FILES)


def frozen_controls(full, residual_old, capacity, previous_guard):
    full_root = full/'predictions'/str(pilot.SEED)
    residual_root = residual_old/'predictions'
    shared = dict(status='PREDICTIONS_FROZEN', seed=pilot.SEED, receivers=list(TARGET), views=list(VIEWS),
                  per_receiver=LIMIT, target_truth_read=False)
    meta = read(full_root/'complete.json')
    if (any(meta.get(k) != value for k, value in shared.items())
            or not {'direct', 'coherent_half', 'complex_response_half'} <= set(meta['modes'])):
        raise ValueError('Unexpected saved Full direct and DURA controls')
    if read(residual_root/'complete.json') != dict(shared, modes=list(residual.MODES), practical=residual.ROUTE):
        raise ValueError('Unexpected saved original residual DURA control')
    references = {}
    for root, method in ((capacity/'predictions', expanded), (previous_guard/'predictions', guard)):
        if read(root/'complete.json') != method.completion():
            raise ValueError('Unexpected matched expanded or guard-half controls')
        references[method.MODES[1]] = root
    return full_root, residual_root, references


def frozen_rows(output, full, residual_old, capacity, previous_guard):
    folder = output/'predictions'
    if read(folder/'complete.json') != pilot.completion():
        raise ValueError('Pilot matrix is not fully frozen')
    if {p.relative_to(folder).as_posix() for p in folder.rglob('*.npy')} != pilot.expected_files():
        raise ValueError('Pilot prediction index is incomplete')
    smoke = read(output/'smoke.json')
    if smoke['status'] != 'PASS' or smoke['target_reads'] != 0 or smoke['seeds'] != list(pilot.SEEDS):
        raise ValueError('Missing real checkpoint no-target check')
    full_root, residual_root, references = frozen_controls(full, residual_old, capacity, previous_guard)
    result = {}
    for route in pilot.ROUTES:
        partial = folder/route/str(pilot.SEED)
        baseline = full_root if route == 'full' else residual_root
        mode = 'coherent_half' if route == 'full' else 'dura_half'
        predicted = scoring.predictions(partial, 'direct', pilot.PER_RECEIVER)
        original = scoring.predictions(baseline, 'direct')[..., :pilot.PER_RECEIVER]
        if not np.array_equal(original, predicted):
            raise ValueError(f'Archived and matched direct decisions differ for {route}')
        row = dict(direct=predicted, dura_half=scoring.predictions(baseline, mode)[..., :pilot.PER_RECEIVER])
        if route == 'full':
            row['complex_response_half'] = scoring.predictions(baseline, 'complex_response_half')[..., :pilot.PER_RECEIVER]
        for ref_mode, root in references.items():
            reference = root/route/str(pilot.SEED)
            if not np.array_equal(original, scoring.predictions(reference, 'direct')[..., :pilot.PER_RECEIVER]):
                raise ValueError(f'Saved {ref_mode} direct decisions differ for {route}')
            row[ref_mode] = scoring.predictions(reference, ref_mode)[..., :pilot.PER_RECEIVER]
        row[pilot.MODE] = scoring.predictions(partial, pilot.MODE, pilot.PER_RECEIVER)
        result[route] = {k: v[None] for k, v in row.items()}
    return result


def score(output, full, residual_old, capacity, previous_guard, dataset):
    rows = frozen_rows(output, full, residual_old, capacity, previous_guard)
    reference = read(output/'source-promotion.json')
    if reference['target_reads'] or set(reference['pairs']) != set(pilot.ROUTES):
        raise ValueError('Missing independent source-only qualification record')
    truth = []
    with zipfile.ZipFile(dataset/'multiple_receiver_test.zip') as archive:
        for i, rx in enumerate(TARGET):
            with recording(archive, f'{rx}_test.h5', output) as handle:
                physical = packet_indices(LIMIT, i)[:pilot.PER_RECEIVER]
                truth.append(label_ids(handle, LIMIT)[physical])
    summary = scoring.score_arrays(rows, np.stack(truth), pilot.MODE, detailed_pairing=True, seeds=pilot.SEEDS)
    write(output/'scores.json', dict(status='TARGET_SCORED', truth_last=True, scope='single_seed_pilot',
                                    seed=pilot.SEED, per_receiver=pilot.PER_RECEIVER,
                                    bootstrap_replicates=scoring.REPLICATES, **summary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for action, args in ((collect, ('receipt', 'config', 'output')),
                         (score, ('output', 'full', 'residual-old', 'capacity', 'previous-guard', 'dataset'))):
        cmd = commands.add_parser(action.__name__)
        for arg in args:
            cmd.add_argument('--'+arg, type=Path, required=True)
        cmd.set_defaults(action=action)
    arguments = vars(parser.parse_args())
    arguments.pop('command')
    arguments.pop('action')(**arguments)
