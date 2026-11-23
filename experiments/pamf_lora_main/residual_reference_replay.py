"""Reuse original LoRa residual R69 and add the two missing frozen Tri-set readers."""

import argparse
from functools import partial
from io import BytesIO
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import tarfile
from time import perf_counter
import zipfile

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_unrolled_restore.wisig_short_training_source import select_gpu
from .collect import wait_for_predictions
from .data import CLASSES, label_ids, packet_indices, recording
from .packet_physics import chirp_windows
from .publish import release, release_commit
from .refine_runner import start
from .residual_dura import LIMIT, ROUTE, RUN as PARENT, SEED as OLD_SEED, TARGET, VIEWS, frozen_models, frozen_target, source_contract, source_role
from .residual_dura import MODES as HISTORICAL_MODES
from .residual_dura_score import prediction
from .runner import activate
from .scoring import macro_f1


RUN = '20261123-phase1-lora-residual-frozen-r69-replay-m2-r01'
SEEDS = (2026092701, 2026092703)
MODES = ('direct', 'dura_half')
GROUPS = {str(k): tuple(TARGET[k::4]) for k in range(4)}


def index_for(seeds, receivers):
    return {f'{s}/{r}__{v}__{m}.npy' for s in seeds for r in receivers for v in VIEWS for m in MODES}


def completion():
    return dict(status='PREDICTIONS_FROZEN', parent_run=PARENT, seeds=list(SEEDS), G_seed=OLD_SEED,
                receivers=list(TARGET), per_receiver=LIMIT, views=list(VIEWS), modes=list(MODES),
                practical=ROUTE, target_truth_read=False)


@torch.no_grad()
def predict(rt, group, *, smoke=False):
    parent = Path(rt['parent_root'])
    if parent.name != PARENT:
        raise ValueError('Expected the original frozen residual LoRa reference')
    device = 'cuda:0'
    g, readers, projector = frozen_models(rt, parent, SEEDS, device)
    root = Path(rt['run_root'])

    def outputs(batch):
        flat = batch.flatten(0, 1)
        projected = projector.restore(flat, g(flat)).reshape_as(batch)
        restored = batch + .5 * (projected - batch)
        result = {s: [f(x)[0] for x in (batch, restored)] for s, f in readers.items()}
        if any(x.shape != (len(batch), len(CLASSES)) or not torch.isfinite(x).all()
               for decisions in result.values() for x in decisions):
            raise ValueError('Frozen LoRa default head failed')
        return result

    if smoke:
        source = Path(rt['source_root'])
        iq, _, _, _ = source_role(source, source_contract(source), 'L_s')
        sample = np.stack([chirp_windows(iq[k]) for k in range(2)])
        outputs(tensor(sample, device))
        write(root / 'smoke.json', dict(status='PASS', real_source_packets=len(sample), G_seed=OLD_SEED,
                                       reader_seeds=list(SEEDS), target_reads=0, truth_reads=0))
        return

    target = frozen_target(parent)
    destination = root / 'predictions'
    elapsed = {}
    for receiver in GROUPS[group]:
        started = perf_counter()
        iq = np.load(target / f'{receiver}.npy', mmap_mode='r', allow_pickle=False)
        if iq.shape != (len(VIEWS), LIMIT, 6, 2, 256) or iq.dtype != np.float32:
            raise ValueError('Unexpected original residual target observations')
        for scene, view in zip(VIEWS, iq, strict=True):
            rows = {seed: [[], []] for seed in SEEDS}
            for batch in tensor(view, device).split(48):
                for seed, result in outputs(batch).items():
                    for row, logits in zip(rows[seed], result, strict=True):
                        row.extend(logits.argmax(1).tolist())
            for seed, result in rows.items():
                for mode, labels in zip(MODES, result, strict=True):
                    with (destination / str(seed) / f'{receiver}__{scene}__{mode}.npy').open('xb') as stream:
                        np.save(stream, np.asarray(labels, dtype=np.int16), allow_pickle=False)
            print('LORA_R69_PAIRED_VIEW', receiver, scene, LIMIT, flush=True)
        elapsed[receiver] = perf_counter() - started
    write(root / f'group-{group}-complete.json', dict(status='COMPLETE', wall_seconds=elapsed, truth_reads=0))


def dispatch(rt):
    root = Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    path = root / 'state.json'
    worker = partial(start, module=rt['runner'])
    try:
        write(path, dict(status='SMOKE'))
        if worker(rt, 'smoke', 'source', select_gpu()).wait():
            raise RuntimeError('Real no-query LoRa reference smoke failed')
        for seed in SEEDS:
            (root / 'predictions' / str(seed)).mkdir(parents=True, exist_ok=False)
        write(path, dict(status='PREDICT'), exclusive=False)
        assigned = []
        procs = []
        for group in GROUPS:
            gpu = select_gpu(exclude=assigned)
            assigned.append(gpu)
            procs.append(worker(rt, 'evaluate', group, gpu))
        codes = [p.wait() for p in procs]
        if any(codes) or {p.relative_to(root/'predictions').as_posix() for p in (root/'predictions').rglob('*.npy')} != index_for(SEEDS, TARGET):
            raise RuntimeError('Complete frozen readout required for both remaining seeds')
        write(root / 'predictions' / 'complete.json', completion())
        write(path, dict(status='PREDICTIONS_FROZEN', target_truth_read=False), exclusive=False)
    except Exception as error:
        write(path, dict(status='FAILED', error=str(error)), exclusive=False)
        raise


def export(ref, ssh_config, output):
    receipt = read(ref)
    run = PurePosixPath(receipt['run_root'])
    if run.name != RUN or receipt['archive_transfer'] != 'VERIFIED':
        raise ValueError('Wrong LoRa frozen baseline release')
    ssh = ['ssh', '-F', str(ssh_config), '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'N607']
    wait_for_predictions(ssh, shlex.quote(str(run)))
    payload = subprocess.check_output([*ssh, 'tar -czf - -C ' + shlex.quote(str(run)) + ' predictions smoke.json'], timeout=120)
    expected = {'predictions/' + p for p in index_for(SEEDS, TARGET)} | {'predictions/complete.json', 'smoke.json'}
    with tarfile.open(fileobj=BytesIO(payload), mode='r:gz') as archive:
        files = {m.name: m for m in archive if m.isfile()}
        if set(files) != expected:
            raise ValueError('Transferred readout coverage differs')
        output.mkdir(parents=True, exist_ok=False)
        for name in expected:
            path = output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(files[name]) as source, path.open('xb') as dest:
                dest.write(source.read())


def score(out, old, dataset):
    expected = dict(status='PREDICTIONS_FROZEN', seed=OLD_SEED, receivers=list(TARGET), per_receiver=LIMIT,
                    views=list(VIEWS), modes=list(HISTORICAL_MODES), practical=ROUTE, target_truth_read=False)
    if read(out / 'predictions' / 'complete.json') != completion() or read(old / 'predictions' / 'complete.json') != expected:
        raise ValueError('Truth remains closed until all three frozen readouts match')
    rows = {}
    for seed in (OLD_SEED, *SEEDS):
        path = old / 'predictions' if seed == OLD_SEED else out / 'predictions' / str(seed)
        rows[seed] = {v: {m: np.stack([prediction(path / f'{rx}__{v}__{m}.npy') for rx in TARGET])
                          for m in MODES} for v in VIEWS}
    with zipfile.ZipFile(dataset / 'multiple_receiver_test.zip') as archive:
        truth = []
        for i, rx in enumerate(TARGET):
            with recording(archive, f'{rx}_test.h5', out) as handle:
                truth.append(label_ids(handle, LIMIT)[packet_indices(LIMIT, i)])
    truth = np.stack(truth)
    records = []
    deltas = []
    for seed, views in sorted(rows.items()):
        measures = []
        for view, modes in views.items():
            hits = [modes[key] == truth for key in MODES]
            raw, dura = hits
            repaired, damaged = dura & ~raw, raw & ~dura
            measures.append(dura.astype(np.int8) - raw)
            for receiver, selection in [('ALL', slice(None)), *zip(TARGET, range(len(TARGET)), strict=True)]:
                selected = truth[selection].ravel()
                inputs = {k: dict(correct=int((v[selection].ravel() == selected).sum()),
                                  macro_f1=macro_f1(selected, v[selection].ravel())) for k, v in modes.items()}
                records.append(dict(seed=seed, view=view, receiver=receiver, count=len(selected), result=inputs,
                                    repaired=int(repaired[selection].sum()), damaged=int(damaged[selection].sum())))
        deltas.append(np.stack(measures))
    physical = np.asarray(deltas, dtype=np.int16).sum(axis=0)
    quantities = np.concatenate((physical, physical[1:].sum(axis=0, keepdims=True)))
    rng = np.random.default_rng(20261123)
    pooled = np.zeros((1600, len(quantities)), dtype=np.int64)
    for phase in range(len(TARGET)):
        sampled = rng.integers(0, LIMIT, size=(1600, LIMIT))
        for scene, x in enumerate(quantities):
            pooled[:, scene] += x[phase][sampled].sum(axis=1)
    bounds = {v: np.quantile(pooled[:, i], (.025, .975)).tolist() for i, v in enumerate((*VIEWS, 'six_scene'))}
    write(out/'scores.json', dict(status='ANALYZED', records=records, paired_ci_net_correct=bounds,
          physical_bootstrap_replicates=len(pooled), truth_last=True, reader_seeds=sorted(rows), G_seed=OLD_SEED))


def publish(repo, output, context, ssh_config, parent_submit):
    config = read(context)
    rt = {key: config[key] for key in ('project', 'python', 'frozen_sources')}
    project = PurePosixPath(rt['project'])
    receipt = read(parent_submit)
    parent = PurePosixPath(receipt['run_root'])
    if parent != project / 'runs' / PARENT or receipt['archive_transfer'] != 'VERIFIED':
        raise ValueError('Expected the original residual LoRa release')
    rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(repo), parent_root=str(parent),
              run_root=str(project / 'runs' / RUN), release_root=str(project / 'releases' / f'{RUN}-release'),
              source_root=str(project / 'releases/20261122-phase1-lora-full-aligned-data-r01-release/data/source'))
    release(repo, output, rt, ssh_config, archive_extras=('tools/r17_activation_factorization.py',))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    steps = p.add_subparsers(dest='step', required=True)
    for action, names in {'publish': ('repo','output','context','ssh-config','parent-submit'),
                          'score': ('receipt','ssh-config','out','old','dataset')}.items():
        cmd = steps.add_parser(action)
        for name in names:
            cmd.add_argument('--'+name, type=Path, required=True)
    for action in ('dispatch', 'evaluate', 'smoke'):
        cmd = steps.add_parser(action)
        cmd.add_argument('runtime', type=Path)
        cmd.add_argument('group', nargs='?')
    c = p.parse_args()
    if c.step == 'publish':
        publish(c.repo, c.output, c.context, c.ssh_config, c.parent_submit)
    elif c.step == 'score':
        if not (c.dataset / 'multiple_receiver_test.zip').is_file():
            raise FileNotFoundError('Original LoRa truth archive required by CPU scorer')
        export(c.receipt, c.ssh_config, c.out)
        score(c.out, c.old, c.dataset)
    else:
        rt = read(c.runtime)
        if c.step == 'dispatch':
            dispatch(rt)
        else:
            activate(rt)
            predict(rt, c.group, smoke=c.step == 'smoke')


if __name__ == '__main__':
    main()
