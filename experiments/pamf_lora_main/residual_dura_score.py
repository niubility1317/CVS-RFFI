"""Independently score the complete frozen residual LoRa matrix."""
import argparse
from io import BytesIO
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import tarfile
import zipfile

import numpy as np

from experiments.cric_dura_comparison.io import read, write
from .collect import wait_for_predictions
from .data import CLASSES, TARGET, VIEWS, label_ids, packet_indices, recording
from .full_packet_data import LIMIT
from .residual_dura import MODES, ROUTE, RUN, SEED
from .scoring import macro_f1


def prediction(path):
    a = np.load(path, allow_pickle=False)
    if (a.shape != (LIMIT,) or a.dtype.kind not in 'iu'
            or a.min() < 0 or a.max() >= len(CLASSES)):
        raise ValueError(f'Invalid LoRa prediction {path.name}')
    return a


def collect(submit, ssh_config, dataset, output, full_control, disturbance_gate=False):
    receipt = read(submit)
    run = PurePosixPath(receipt['run_root'])
    allowed, modes = {RUN: SEED}, MODES
    if disturbance_gate:
        from .residual_dura_gate_run import MODES as gate_modes, SEED_RUNS
        allowed, modes = {name: seed for seed, name in SEED_RUNS.items()}, gate_modes
    if run.name not in allowed or receipt['archive_transfer'] != 'VERIFIED':
        raise ValueError('Wrong residual DURA receipt')
    seed = allowed[run.name]
    root = shlex.quote(str(run))
    conn = ['ssh', '-F', str(ssh_config), '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'N607']
    wait_for_predictions(conn, root)
    extra = ' activity.json' if disturbance_gate else ''
    payload = subprocess.check_output([*conn, f'tar -czf - -C {root} predictions smoke.json{extra}'], timeout=180)
    output.mkdir(parents=True, exist_ok=False)
    p = output / 'predictions'
    p.mkdir()
    matrix = {f'{r}__{v}__{m}.npy' for r in TARGET for v in VIEWS for m in modes}
    allowed = {'smoke.json', 'predictions/complete.json', *('predictions/'+x for x in matrix)}
    if disturbance_gate:
        allowed.add('activity.json')
    with tarfile.open(fileobj=BytesIO(payload), mode='r:gz') as archive:
        for member in archive:
            if member.isdir() and member.name == 'predictions':
                continue
            if not member.isfile() or member.name not in allowed:
                raise ValueError('Unexpected residual DURA scoring artifact')
            with archive.extractfile(member) as src, (output / member.name).open('xb') as dst:
                dst.write(src.read())
    if read(p / 'complete.json') != dict(status='PREDICTIONS_FROZEN', seed=seed,
            receivers=list(TARGET), per_receiver=LIMIT, views=list(VIEWS), modes=list(modes),
            practical=ROUTE, target_truth_read=False) or {f.name for f in p.glob('*.npy')} != matrix:
        raise ValueError('Residual LoRa predictions were not all frozen before scoring')
    rows = {(r, v, m): prediction(p / f'{r}__{v}__{m}.npy')
            for r in TARGET for v in VIEWS for m in modes}
    coverage = {}
    if disturbance_gate:
        smoke = read(output / 'smoke.json')
        if (smoke['status'] != 'PASS' or smoke['stage'] != 'r69' or smoke['seed'] != seed
                or smoke['F_reads'] or smoke['TX_labels_read'] or smoke['target_reads']):
            raise ValueError('Mismatched independent gate source smoke')
        coverage = read(output / 'activity.json')
        if set(coverage) != set(TARGET) or any(set(item) != set(VIEWS) for item in coverage.values()):
            raise ValueError('Incomplete residual disturbance gate activity')
        for r in TARGET:
            for v in VIEWS:
                count = coverage[r][v]
                if type(count) is not int or not 0 <= count <= LIMIT:
                    raise ValueError('Invalid residual gate activity count')
                if count in (0, LIMIT):
                    reference = 'dura_half' if count else 'direct'
                    if not np.array_equal(rows[r, v, 'dura_gated_half'], rows[r, v, reference]):
                        raise ValueError('Gated predictions disagree with uniform gate activity')
    full = full_control / 'predictions' / str(seed)
    control = read(full / 'complete.json')
    if (control['status'] != 'PREDICTIONS_FROZEN' or control['seed'] != seed
            or control['receivers'] != list(TARGET) or control['per_receiver'] != LIMIT
            or control['views'] != list(VIEWS) or control['target_truth_read'] is not False):
        raise ValueError('Full practical prediction control does not match')
    full_modes = ('direct', 'complex_response_half')
    full_rows = {(r, v, m): prediction(full / f'{r}__{v}__{m}.npy')
                 for r in TARGET for v in VIEWS for m in full_modes}
    clean_control_changes = sum(int((rows[r, 'clean', 'direct'] != full_rows[r, 'clean', 'direct']).sum())
                                for r in TARGET)
    truth = {}
    with zipfile.ZipFile(dataset / 'multiple_receiver_test.zip') as archive:
        for i, rx in enumerate(TARGET):
            with recording(archive, f'{rx}_test.h5', output) as h5:
                truth[rx] = label_ids(h5, LIMIT)[packet_indices(LIMIT, i)]
    answer = np.concatenate([truth[r] for r in TARGET])
    metrics = {}
    for m, source_rows in [*((m, rows) for m in modes),
                           *((f'full_{m}', full_rows) for m in full_modes)]:
        key = m.removeprefix('full_') if source_rows is full_rows else m
        views = {}
        for v in VIEWS:
            pred = np.concatenate([source_rows[r, v, key] for r in TARGET])
            direct = np.concatenate([rows[r, v, 'direct'] for r in TARGET]) == answer
            ok = pred == answer
            views[v] = dict(correct=int(ok.sum()), n=len(ok), accuracy=float(ok.mean()),
                            macro_f1=macro_f1(answer, pred), gained=int((ok&~direct).sum()),
                            lost=int((~ok&direct).sum()),
                            by_rx=dict(zip(TARGET, ok.reshape(len(TARGET), LIMIT).sum(1).tolist(),
                                           strict=True)))
        practical = sum(views[v]['correct'] for v in VIEWS[1:])
        average = practical / (len(VIEWS[1:])*len(answer))
        metrics[m] = dict(views=views, practical_correct=practical, practical_mean=average,
                          clean_gap=views['clean']['accuracy']-average)
    summary = dict(status='SCORED_COMPLETE', truth_last=True, seed=seed, practical=ROUTE,
                   clean_control_prediction_changes=clean_control_changes,
                   matrix_arrays=len(rows), metrics=metrics)
    if disturbance_gate:
        summary['gate_activity'] = coverage
    write(output / 'scores.json', summary)
    print('RESIDUAL_DURA_SCORED', output / 'scores.json', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('submit', 'ssh-config', 'dataset', 'output', 'full-control'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--disturbance-gate', action='store_true')
    collect(**vars(parser.parse_args()))
