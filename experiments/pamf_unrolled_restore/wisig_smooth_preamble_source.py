"""Verify fixed smooth preamble timing with the registered WiSig source reader."""

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
from . import wisig_short_training_source as core
from .wisig_joint_preamble_source import reader_factory
from .wisig_smooth_preamble import MODES as READOUTS, observations, readouts
from .wisig_uncertain_source import source_iq


RUN = '20261124-phase1-wisig-smooth-preamble-source-m2-r01'
MODES = (*READOUTS, 'r69')
COMPARISONS = (('smooth_full', 'fixed_three'), ('smooth_held', 'smooth_full'),
               ('smooth_held', 'smooth_held_reflect'), ('smooth_held_reflect', 'fixed_three_reflect'),
               ('smooth_held', 'fixed_three'), ('smooth_held', 'direct'), ('smooth_held', 'r69'))


def calibrate(source_root, output):
    source, _, indices = source_iq(source_root)
    margins = {}
    started = perf_counter()
    for part in np.array_split(indices, len(indices)//64):
        _, _, fits = observations(source[part])
        for name, (_, margins_part, _) in fits.items():
            margins[name] = max(margins.get(name, -np.inf), float(margins_part.max()))
    thresholds = {name: maximum+.02*abs(maximum)+1e-6 for name, maximum in margins.items()}
    write(output, dict(thresholds=thresholds, count=len(indices), trainable_parameters=0, optimizer_updates=0,
                       seconds=perf_counter()-started, source_V_reads=0, F_reads=0, TX_reads=0, target_reads=0))


@torch.no_grad()
def smoke(rt, thresholds):
    source, _, calibration = source_iq(Path(rt['source_root']))
    quiet = source[calibration[:12]]
    reader = reader_factory(rt).eval().requires_grad_(False)
    variants, _ = readouts(quiet, thresholds)
    if tuple(variants) != READOUTS or any(not np.array_equal(quiet, x) for x in variants.values()):
        raise ValueError('Clean calibration IQ must bypass all frozen timing readouts')
    disturbed = quiet[..., np.minimum(np.arange(256)+28, 482-np.arange(256))]
    noisy, _ = readouts(disturbed, thresholds)
    if np.array_equal(noisy['smooth_held'], disturbed):
        raise ValueError('Controlled sample shift did not exercise smooth timing')
    restore = core.original_r69(Path(rt['parent_root']))
    outputs = [tensor(x, 'cuda:0') for x in noisy.values()]
    outputs.extend(restore(tensor(quiet, 'cuda:0')))
    for y in outputs:
        logits = reader.classify_features(reader.features(y))
        if logits.shape != (len(y), 6) or not torch.isfinite(logits).all():
            raise ValueError('Matched real reader requires finite [B, 6] decisions')
    write(Path(rt['run_root']) / 'smoke.json', dict(status='PASS', clean=len(quiet), modes=list(MODES), target_reads=0))


def dispatch(rt):
    root = Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    try:
        write(root/'state.json', dict(status='CALIBRATE'))
        calibrate(Path(rt['source_root']), root/'threshold.json')
        for action in ('smoke', 'evaluate', 'score'):
            gpu = None if action == 'score' else core.select_gpu()
            write(root/'state.json', dict(status=action.upper(), gpu=gpu), exclusive=False)
            worker = core.start(rt, action, 'smooth-preamble', gpu, module=rt['runner'], extra_keys=('recognizer_run',))
            if worker.wait():
                raise RuntimeError(f'WiSig smooth preamble {action} failed')
        write(root/'state.json', dict(status='SOURCE_SCORED'), exclusive=False)
    except Exception as exc:
        write(root/'state.json', dict(status='FAILED', error=str(exc)), exclusive=False)
        raise


def publish(repo, output, context, ssh_config):
    cfg = read(context)
    rt = {key: cfg[key] for key in ('project', 'python', 'frozen_sources')}
    project = PurePosixPath(rt['project'])
    runs = project/'runs'
    rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(repo),
              release_root=str(project/'releases'/f'{RUN}-release'), run_root=str(runs/RUN),
              source_root=str(runs/core.CACHE_RUN/'cache'), parent_root=str(runs/core.R17_RUN),
              recognizer_run=str(runs/tri_plan.RUN))
    release(repo, output, rt, ssh_config, archive_extras=('tools/r17_activation_factorization.py',))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    pub = actions.add_parser('publish')
    for name in ('repo', 'output', 'context', 'ssh-config'):
        pub.add_argument('--' + name, type=Path, required=True)
    for action in ('dispatch', 'smoke', 'evaluate', 'score'):
        item = actions.add_parser(action)
        item.add_argument('runtime', type=Path)
        item.add_argument('name', nargs='?')
    args = parser.parse_args()
    if args.action == 'publish':
        publish(args.repo, args.output, args.context, args.ssh_config)
    else:
        rt = read(args.runtime)
        if args.action == 'dispatch':
            dispatch(rt)
        elif args.action == 'score':
            core.score(rt, MODES, comparisons=COMPARISONS)
        else:
            activate(rt)
            thresholds = read(Path(rt['run_root'])/'threshold.json')['thresholds']
            if args.action == 'smoke':
                smoke(rt, thresholds)
            else:
                core.predict(rt, transform=partial(readouts, thresholds=thresholds), readout_names=READOUTS,
                             reader_factory=reader_factory)
