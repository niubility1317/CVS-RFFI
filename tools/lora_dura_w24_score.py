"""Collect the complete W24 pilot matrix and score it independently on the truth side."""

import argparse
import io
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import tarfile
import zipfile

import numpy as np

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main.data import TARGET, VIEWS, label_ids, packet_indices, recording
from experiments.pamf_lora_main import dura_w24_augmented_target as augmented
from experiments.pamf_lora_main.dura_w24_target import MODES, N, RUN, SHARDS, completion, prediction_names
from experiments.pamf_lora_main.full_packet_data import LIMIT


REPLICATES = 1600


def collect(receipt, config, output):
    record = read(receipt)
    root = PurePosixPath(record['run_root'])
    if root.name not in (RUN, augmented.RUN) or record['archive_transfer'] != 'VERIFIED':
        raise ValueError('Expected the registered W24 pilot receipt')
    ssh = ['ssh', '-F', str(config), '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'N607']
    state = subprocess.check_output([*ssh, 'cat ' + shlex.quote(str(root / 'state.json'))], timeout=20)
    if json.loads(state)['status'] != 'PREDICTIONS_FROZEN':
        raise ValueError('All W24 predictions must freeze before collection')
    rows = [f'shard-{i}.json' for i in range(SHARDS)]
    command = f'tar -czf - -C {shlex.quote(str(root))} predictions smoke.json ' + ' '.join(rows)
    payload = subprocess.check_output([*ssh, command], timeout=60)
    expected = {'smoke.json', 'predictions/complete.json', *rows,
                *(f'predictions/{name}' for name in prediction_names())}
    with tarfile.open(fileobj=io.BytesIO(payload), mode='r:gz') as archive:
        members = {row.name: row for row in archive if row.isfile()}
        if members.keys() != expected:
            raise ValueError('Unexpected W24 prediction archive')
        output.mkdir(parents=True, exist_ok=False)
        (output / 'predictions').mkdir()
        for name in expected:
            with archive.extractfile(members[name]) as source, (output / name).open('xb') as dest:
                dest.write(source.read())


def frozen(output, *, augmented_reference=None):
    prediction_dir = output / 'predictions'
    is_augmented = augmented_reference is not None
    expected = augmented.completion() if is_augmented else completion()
    modes = augmented.MODES if is_augmented else MODES
    n = LIMIT if is_augmented else N
    if (read(prediction_dir / 'complete.json') != expected
            or read(output / 'smoke.json')['target_reads'] != 0
            or {p.name for p in prediction_dir.glob('*.npy')} != prediction_names()):
        raise ValueError('W24 prediction matrix was not legally frozen')
    rows = [read(output / f'shard-{i}.json') for i in range(SHARDS)]
    if any(r['status'] != 'COMPLETE' or r['truth_reads'] for r in rows):
        raise ValueError('Invalid W24 worker completion')
    if is_augmented and read(augmented_reference / 'complete.json') != augmented.reference_completion():
        raise ValueError('Missing the same-reader archived augmented W24 reference matrix')
    result = []
    for i, view in enumerate(VIEWS):
        pred = np.stack([np.load(prediction_dir / f'{rx}__{view}.npy', allow_pickle=False) for rx in TARGET])
        if (pred.shape != (len(TARGET), len(modes), n if i else LIMIT)
                or pred.dtype != np.int16 or pred.min() < 0 or pred.max() > 9):
            raise ValueError(f'Invalid W24 blind predictions for {view}')
        if is_augmented:
            reference = np.stack([[np.load(augmented_reference / f'{rx}__{view}__{mode}.npy', allow_pickle=False)
                                   for mode in augmented.REFERENCE_MODES] for rx in TARGET])
            if (reference.shape != pred.shape or reference.dtype != np.int16
                    or reference.min() < 0 or reference.max() > 9
                    or not np.array_equal(reference[:, 0], pred[:, 0])):
                raise ValueError(f'Augmented W24 archived direct decisions differ for {view}')
            pred = np.concatenate((reference, pred[:, 1:]), axis=1)
        result.append(pred)
    return result, (*augmented.REFERENCE_MODES, modes[1]) if is_augmented else modes, n


def paired(left, right):
    repaired, harmed = right & ~left, left & ~right
    change = repaired.astype(np.int16) - harmed.astype(np.int16)
    # Resample physical packets within RX. The same resample covers all six scenes.
    values = change.sum(axis=0)
    rng = np.random.default_rng(20261125)
    draws = np.zeros(REPLICATES, dtype=np.int64)
    for rx_delta in values:
        indices = rng.integers(0, len(rx_delta), (REPLICATES, len(rx_delta)))
        draws += rx_delta[indices].sum(axis=1)
    return dict(repairs=int(repaired.sum()), harms=int(harmed.sum()), net=int(change.sum()),
                ci95=np.quantile(draws, [.025, .975]).tolist(),
                by_rx={name: int(delta.sum()) for name, delta in zip(TARGET, values, strict=True)})


def score(output, dataset, *, augmented_reference=None):
    decisions, modes, n = frozen(output, augmented_reference=augmented_reference)
    truth = []
    with zipfile.ZipFile(dataset / 'multiple_receiver_test.zip') as archive:
        for i, rx in enumerate(TARGET):
            with recording(archive, f'{rx}_test.h5', output) as handle:
                truth.append(label_ids(handle, LIMIT)[packet_indices(LIMIT, i)])
    labels = np.stack(truth)[:, None]
    clean, practical = decisions[0] == labels, np.stack(decisions[1:]) == labels[..., :n]
    if int(clean[:, 0].sum()) != (28566 if augmented_reference is not None else 28146):
        raise ValueError('Independent W24 clean score differs from the archived frozen reader')
    payload = {}
    pairs = {f'{modes[b]}_vs_{modes[a]}': (a, b) for b in range(1, len(modes)) for a in range(b)}
    for view, hits in zip((*VIEWS, 'six_scene'), (clean[None], *(v[None] for v in practical), practical), strict=True):
        payload[view] = dict(n=int(hits[:, :, 0].size), correct=dict(zip(modes, hits.sum(axis=(0, 1, 3)).tolist())),
                             paired={name: paired(hits[:, :, a], hits[:, :, b]) for name, (a, b) in pairs.items()},
                             by_rx={rx: dict(zip(modes, rx_hits.sum(axis=(0, 2)).tolist()))
                                    for rx, rx_hits in zip(TARGET, np.moveaxis(hits, 1, 0), strict=True)})
    write(output / 'scores.json', dict(status='TARGET_SCORED', truth_last=True, views=payload,
                                       replicates=REPLICATES, exact_archived_clean=True,
                                       exact_archived_all_views=augmented_reference is not None))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    for action, fields in ((collect, ('output', 'receipt', 'config')), (score, ('output', 'dataset'))):
        command = commands.add_parser(action.__name__)
        for name in fields:
            command.add_argument(f'--{name}', type=Path, required=True)
        if action == score:
            command.add_argument('--augmented-reference', type=Path)
        command.set_defaults(action=action)
    args = vars(parser.parse_args())
    args.pop('action')(**args)
