"""Train matched recovery with or without an observable timing front end."""

import argparse
from functools import partial
from pathlib import Path, PurePosixPath

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor
from experiments.pamf_lora_main.publish import release, release_commit
from experiments.pamf_lora_main.runner import activate

from . import wisig_uncertain_source as core
from .wisig_timing import PERIOD, estimate_timing, realign
from .wisig_timing_source import RUN as TIMING_RUN
from .wisig_uncertain_prior import as_complex


RUN = '20261123-phase1-wisig-aligned-recovery-source-s2026092701-m2-r01'
RECIPE = {'untracked': .20, 'aligned': .20}


def shifted_inputs(inputs, rng):
    batch, _, length = inputs.shape
    delta = torch.randint(-2 * PERIOD, 2 * PERIOD + 1, (batch, 2), device=inputs.device, generator=rng)
    delta[-4:] = 0  # Retain the same identity exposures as the source training control.
    index = torch.arange(length, device=inputs.device) - delta[:, (0, 1, 0), None]
    mirror = torch.where(index < 0, -index, torch.minimum(index, 2 * (length - 1) - index))
    return inputs.gather(dim=-1, index=mirror).flatten(0, 1)


def estimate_and_align(z, threshold):
    # Fit and inference use the same received-IQ estimator, with no true delay
    # tensor or underlying augmentation state entering this function.
    storage = np.asarray(torch.view_as_real(z).detach().cpu().tolist(), dtype=np.float32)
    x = storage[..., 0] + 1j * storage[..., 1]
    delta, margin, estimates = estimate_timing(x)
    active = margin > threshold
    y = realign(x, np.where(active, delta, 0), protocol=True, cfo=estimates.coarse_rad_per_sample)
    iq = np.stack((y.real, y.imag), axis=1).astype(np.float32, copy=False)
    return as_complex(tensor(iq, z.device)), tensor(active, z.device)


def observe(inputs, rng, arm, threshold):
    rich = shifted_inputs(inputs, rng)
    return rich if arm == 'untracked' else estimate_and_align(rich, threshold)[0]


def prepare(z, arm, threshold):
    return (z, None) if arm == 'untracked' else estimate_and_align(z, threshold)


def clean_calibration(run_root):
    ancestor = Path(run_root).parent / TIMING_RUN
    stored = read(ancestor / 'threshold.json')
    if (stored['count'], stored['quantile']) != (core.CAL_COUNT, .999) or any(
            stored[k] for k in ('TX_reads', 'V_reads', 'F_reads', 'target_reads')):
        raise ValueError('Expected earlier fixed U-only timing calibration')
    if not np.isfinite(stored['threshold']) or stored['threshold'] < 0:
        raise ValueError('Invalid fixed timing margin')
    return dict(stored, provenance=str(ancestor / 'threshold.json'))


def publish(repo, output, context, ssh_config):
    config = read(context)
    rt = {key: config[key] for key in ('project', 'python', 'frozen_sources')}
    project = PurePosixPath(rt['project'])
    rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(repo),
              release_root=str(project / 'releases' / f'{RUN}-release'), run_root=str(project / 'runs' / RUN),
              source_root=str(project / 'runs' / core.baseline.CACHE_RUN / 'cache'),
              parent_root=str(project / 'runs' / core.baseline.R17_RUN),
              baseline_root=str(project / 'runs' / core.baseline.RUN))
    release(repo, output, rt, ssh_config, archive_extras=('tools/r17_activation_factorization.py',))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    pub = actions.add_parser('publish')
    for name in ('repo', 'output', 'context', 'ssh-config'):
        pub.add_argument('--' + name, type=Path, required=True)
    for action in ('dispatch', 'smoke', 'fit', 'evaluate', 'score'):
        worker = actions.add_parser(action)
        worker.add_argument('runtime', type=Path)
        worker.add_argument('name', nargs='?')
    cmd = parser.parse_args()
    if cmd.action == 'publish':
        publish(cmd.repo, cmd.output, cmd.context, cmd.ssh_config)
        return
    rt = read(cmd.runtime)
    if cmd.action == 'dispatch':
        core.dispatch(rt, RECIPE, clean_calibration(rt['run_root']))
    elif cmd.action == 'score':
        core.baseline.score(rt, core.modes_for(RECIPE))
    else:
        threshold = read(Path(rt['run_root']) / 'timing.json')['threshold']
        if cmd.action in ('smoke', 'fit'):
            core.fit(rt, cmd.name, smoke=cmd.action == 'smoke', recipe=RECIPE,
                     observe=partial(observe, arm=cmd.name, threshold=threshold))
        else:
            activate(rt)
            core.evaluate(rt, RECIPE, prepare=partial(prepare, threshold=threshold))


if __name__ == '__main__':
    main()
