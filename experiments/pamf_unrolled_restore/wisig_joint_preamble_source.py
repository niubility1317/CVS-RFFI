"""Compare frozen WiSig preamble engineering controls on paired source V."""

import argparse
from functools import partial
from pathlib import Path, PurePosixPath
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_lora_main.publish import release, release_commit
from experiments.pamf_lora_main.runner import activate
from experiments.pamf_repl_view_shuffle import plan as tri_plan
from experiments.pamf_repl_view_shuffle.source import load_reader

from . import wisig_short_training_source as core
from .wisig_aligned_source import clean_calibration
from .wisig_joint_preamble import READOUTS, TAP_OFFSETS, fit, observable, readouts
from .wisig_uncertain_source import source_iq


RUN = '20261123-phase1-wisig-joint-preamble-source-s2026092701-r01'
SEED = 2026092701
ROW = f'main_split-repl-shuffled_time-3view-s{SEED}'
MODES = (*READOUTS, 'r69')


def reader_factory(rt):
    root = Path(rt['recognizer_run'])
    cfg = read(root / 'configs' / f'predict__{ROW}.json')
    if (cfg['run_id'], cfg['row_id'], cfg['model_seed']) != (tri_plan.RUN, ROW, SEED):
        raise ValueError('Expected the registered WiSig Main split Tri-SIG reader')
    reader, source = load_reader(cfg['source_output'], read(cfg['source_contract']), 'cuda:0')
    if (source['run_id'], source['row_id'], source['model_seed']) != (tri_plan.RUN, ROW, SEED):
        raise ValueError('Unexpected actual frozen reader')
    return reader


def calibrate(source_root, local, output):
    source, _, indices = source_iq(Path(source_root))
    margins = {name: [] for name in TAP_OFFSETS}
    started = perf_counter()
    for chunk in np.array_split(indices, len(indices) // core.BATCH):
        _, _, sync = observable(source[chunk])
        for name in margins:
            margins[name].append(fit(sync, name).margin)
    thresholds = {'local_transition': local['threshold']}
    for name, parts in margins.items():
        thresholds[name] = max(0., float(np.quantile(np.concatenate(parts), .999)))
    write(output, dict(thresholds=thresholds, count=len(indices), quantile=.999,
                       trainable_parameters=0, optimizer_updates=0, calibration_seconds=perf_counter()-started,
                       F_reads=0, V_reads=0, TX_reads=0, target_reads=0))


@torch.no_grad()
def smoke(rt, thresholds):
    source, _, indices = source_iq(Path(rt['source_root']))
    x = source[indices[:4]]
    reader = reader_factory(rt).eval().requires_grad_(False)
    engine = core.original_r69(Path(rt['parent_root']))
    variants, _ = readouts(x, thresholds)
    restored, = engine(tensor(x, 'cuda:0'))
    inputs = [tensor(variants[name], 'cuda:0') for name in READOUTS] + [restored]
    for y in inputs:
        logits = reader.classify_features(reader.features(y))
        if logits.shape != (len(x), 6) or not torch.isfinite(logits).all():
            raise ValueError('Real checkpoint no-query smoke failed')
    write(Path(rt['run_root']) / 'smoke.json', dict(status='PASS', count=len(x), modes=list(MODES), V_reads=0,
                                                 TX_reads=0, target_reads=0))


def dispatch(rt):
    root = Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    status = root / 'state.json'
    try:
        write(status, dict(status='CALIBRATE'))
        calibrate(rt['source_root'], clean_calibration(root), root / 'threshold.json')
        for action in ('smoke', 'evaluate', 'score'):
            gpu = None if action == 'score' else core.select_gpu()
            write(status, dict(status=action.upper(), gpu=gpu), exclusive=False)
            worker = core.start(rt, action, 'joint-preamble', gpu, module=rt['runner'], extra_keys=('recognizer_run',))
            if worker.wait():
                raise RuntimeError(f'WiSig joint-preamble {action} failed')
        write(status, dict(status='SOURCE_SCORED'), exclusive=False)
    except Exception as exc:
        write(status, dict(status='FAILED', error=str(exc)), exclusive=False)
        raise


def publish(repo, output, context, ssh_config):
    config = read(context)
    rt = {key: config[key] for key in ('project', 'python', 'frozen_sources')}
    project = PurePosixPath(rt['project'])
    rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(repo),
              release_root=str(project / 'releases' / f'{RUN}-release'), run_root=str(project / 'runs' / RUN),
              source_root=str(project / 'runs' / core.CACHE_RUN / 'cache'),
              parent_root=str(project / 'runs' / core.R17_RUN), recognizer_run=str(project / 'runs' / tri_plan.RUN))
    release(repo, output, rt, ssh_config, archive_extras=('tools/r17_activation_factorization.py',))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    pub = actions.add_parser('publish')
    for name in ('repo', 'output', 'context', 'ssh-config'):
        pub.add_argument('--' + name, type=Path, required=True)
    for action in ('dispatch', 'smoke', 'evaluate', 'score'):
        worker = actions.add_parser(action)
        worker.add_argument('runtime', type=Path)
        worker.add_argument('name', nargs='?')
    cmd = parser.parse_args()
    if cmd.action == 'publish':
        publish(cmd.repo, cmd.output, cmd.context, cmd.ssh_config)
        return
    rt = read(cmd.runtime)
    if cmd.action == 'dispatch':
        dispatch(rt)
    elif cmd.action == 'score':
        core.score(rt, MODES)
    else:
        activate(rt)
        thresholds = read(Path(rt['run_root']) / 'threshold.json')['thresholds']
        if cmd.action == 'smoke':
            smoke(rt, thresholds)
        else:
            core.predict(rt, transform=partial(readouts, thresholds=thresholds), readout_names=READOUTS,
                         reader_factory=reader_factory)


if __name__ == '__main__':
    main()
