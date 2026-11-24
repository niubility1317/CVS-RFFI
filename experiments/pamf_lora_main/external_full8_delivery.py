"""Release clean-only external comparisons and score frozen decisions independently."""
import argparse
from pathlib import Path, PurePosixPath
import shlex
import zipfile

import numpy as np

from experiments.cric_dura_comparison.io import read, write
from .clean_aperture import NATIVE_EDGE_POINTS
from .collect import receive, wait_for_predictions
from .data import CLASSES, TARGET, label_ids, packet_indices, recording
from .external_full8 import METHODS, NATIVE_EDGE_RUN, RECOVERY_RUN, RUN, SEED
from .full_packet_data import LIMIT
from .full_packet_publish import DATA_RUN
from .publish import release, release_commit
from .scoring import macro_f1


FULL8_CACHE_RUN = '20261125-phase1-lora-pamf-tri-clean-relativetone-m2-r01'
NATIVE_EDGE_CACHE_RUN = '20261125-phase1-lora-pamf-tri-clean-nativeboundary98-s2026092702-r01'


def publish(repo, output, context, ssh_config, recovery=False, native_edge=False):
    if recovery and native_edge:
        raise ValueError('Recovery and W12 scratch training are separate runs')
    ctx = read(context)
    project = PurePosixPath(ctx['project'])
    native = project/'releases'/f'{DATA_RUN}-release'/'data'
    rt = {k: ctx[k] for k in ('project', 'frozen_sources', 'external_releases',
                             'ibrff_model_root', 'sdg_model_root')}
    run_id = RECOVERY_RUN if recovery else NATIVE_EDGE_RUN if native_edge else RUN
    cache_run, aperture = ((NATIVE_EDGE_CACHE_RUN, '384-all8-nativeedges') if native_edge
                           else (FULL8_CACHE_RUN, '384-all8'))
    gpus = [0, 1, 4, 7, 5, 6] if native_edge else [4, 0, 1, 2] if recovery else [0, 1, 2, 4]
    rt.update(run_id=run_id, code_commit=release_commit(repo), runner='experiments.pamf_lora_main.external_full8',
              python=ctx['external_python'], release_root=str(project/'releases'/f'{run_id}-release'),
              run_root=str(project/'runs'/run_id), source_root=str(native/'source'),
              native_target_root=str(native/'target'),
              clean_cache=str(project/'runs'/cache_run/'clean_cache'/aperture),
              allowed_gpus=gpus,
              min_free_mib=8192, sdg_min_free_mib=10240,
              max_compute_processes=12)
    release(repo, output, rt, ssh_config)


def frozen_predictions(output, names, run_id=RUN):
    if tuple(names) != METHODS:
        raise ValueError('Clean-only scorer requires each registered external row')
    result = {}
    for method in names:
        folder = output/'predictions'/method
        declared = dict(status='PREDICTIONS_FROZEN', method=method, seed=SEED, views=['clean'],
                        receivers=list(TARGET), per_receiver=LIMIT, target_truth_read=False)
        if run_id == NATIVE_EDGE_RUN:
            declared.update(run_id=NATIVE_EDGE_RUN, native_edge_points=NATIVE_EDGE_POINTS)
        if read(folder/'complete.json') != declared:
            raise ValueError(f'Incomplete external clean predictions: {method}')
        chunks = []
        for rx in TARGET:
            path = folder/f'{rx}__clean__direct.npy'
            row = np.load(path, allow_pickle=False)
            if row.shape != (LIMIT,) or row.dtype.kind not in 'iu' or (row < 0).any() or (row >= len(CLASSES)).any():
                raise ValueError(f'Malformed external clean predictions: {method}/{rx}')
            chunks.append(row)
        result[method] = np.concatenate(chunks)
    return result


def score(submit, ssh_config, dataset, output):
    run_root = PurePosixPath(read(submit)['run_root'])
    if not run_root.is_absolute() or run_root.name not in (RUN, RECOVERY_RUN, NATIVE_EDGE_RUN):
        raise ValueError('Wrong external Full8 prediction run')
    root = shlex.quote(str(run_root))
    conn = ['ssh', '-F', str(ssh_config), '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'N607']
    state = wait_for_predictions(conn, root)
    source_run_id = RUN if run_root.name == RECOVERY_RUN else run_root.name
    if run_root.name != RUN and state.get('source_run') != source_run_id:
        raise ValueError('External predictions do not declare the expected source training run')
    output.mkdir(parents=True, exist_ok=False)
    receive(conn, root, output, main=False)
    preds = frozen_predictions(output, state['names'], run_root.name)
    truth = []
    with zipfile.ZipFile(dataset/'multiple_receiver_test.zip') as archive:
        for i, rx in enumerate(TARGET):
            with recording(archive, f'{rx}_test.h5', output) as h5:
                truth.append(label_ids(h5, LIMIT)[packet_indices(LIMIT, i)])
    truth = np.concatenate(truth)
    results = {}
    for method, pred in preds.items():
        correct = pred == truth
        rx_accuracy = correct.reshape(len(TARGET), LIMIT).mean(1)
        results[method] = dict(n=len(truth), correct=int(correct.sum()), accuracy=float(correct.mean()),
                               macro_f1=macro_f1(truth, pred),
                               by_rx={rx: float(rx_accuracy[i]) for i, rx in enumerate(TARGET)},
                               by_tx={str(tx): float(correct[truth == i].mean()) for i, tx in enumerate(CLASSES)})
    write(output/'scores.json', dict(status='SCORED_COMPLETE', run_id=run_root.name, source_run_id=source_run_id,
                                     seed=SEED, truth_last=True, views=['clean'], results=results))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    actions = parser.add_subparsers(dest='kind', required=True)
    for command, fields in (('publish', ('repo', 'output', 'context', 'ssh-config')),
                            ('score', ('submit', 'output', 'dataset', 'ssh-config'))):
        action = actions.add_parser(command)
        for field in fields:
            action.add_argument(f'--{field}', type=Path, required=True)
        if command == 'publish':
            mode = action.add_mutually_exclusive_group()
            mode.add_argument('--recovery', action='store_true')
            mode.add_argument('--native-edge', action='store_true')
    arguments = vars(parser.parse_args())
    kind = arguments.pop('kind')
    {'publish': publish, 'score': score}[kind](**arguments)
