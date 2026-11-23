"""Source-only timing calibration and truth-last same-waveform comparison."""

import argparse
from functools import partial
from pathlib import Path, PurePosixPath

import numpy as np

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main.publish import release, release_commit
from experiments.pamf_lora_main.runner import activate

from . import wisig_short_training_source as core
from .wisig_timing import READOUTS, estimate_timing, readouts
from .wisig_uncertain_source import source_iq


RUN = '20261123-phase1-wisig-timing-engineering-source-s2026092701-r01'
MODES = (*READOUTS, 'r69')


def calibrate(source_root, output):
    x, _, positions = source_iq(Path(source_root))
    margins = []
    for rows in np.array_split(positions, len(positions) // core.BATCH):
        iq = x[rows]
        _, margin, _ = estimate_timing(iq[:, 0] + 1j * iq[:, 1])
        margins.append(margin)
    value = max(0., float(np.quantile(np.concatenate(margins), .999)))
    write(output, dict(threshold=value, count=len(positions), quantile=.999, TX_reads=0,
                       V_reads=0, F_reads=0, target_reads=0))


def dispatch(rt):
    root = Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    status = root / 'state.json'
    try:
        write(status, dict(status='CALIBRATE'))
        calibrate(rt['source_root'], root / 'threshold.json')
        for action in ('smoke', 'evaluate', 'score'):
            gpu = None if action == 'score' else core.select_gpu()
            write(status, dict(status=action.upper(), gpu=gpu), exclusive=False)
            if core.start(rt, action, 'timing', gpu, module=rt['runner']).wait():
                raise RuntimeError(f'WiSig timing {action} failed')
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
              parent_root=str(project / 'runs' / core.R17_RUN))
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
        threshold = read(Path(rt['run_root']) / 'threshold.json')['threshold']
        core.predict(rt, smoke=cmd.action == 'smoke', transform=partial(readouts, threshold=threshold), readout_names=READOUTS)


if __name__ == '__main__':
    main()
