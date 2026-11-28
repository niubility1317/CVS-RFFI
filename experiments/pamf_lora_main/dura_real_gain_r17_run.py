"""Launch and score the next source-only LoRa R17 fusion candidate."""

import argparse
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main.dura_real_gain_fit import LORA_ACTION, LORA_GAIN
from experiments.pamf_lora_main.dura_real_gain_run import free_gpus


RUN = '20261128-phase1-lora-r17-realgain-e104-s2026112861-r01'


def dispatch(runtime):
    rt = read(runtime)
    root = Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    state = root / 'state.json'
    try:
        devices = free_gpus()
        if not devices:
            raise RuntimeError('No GPU with 8 GiB free at launch')
        # Keep the concurrent WiSig R17 worker's physical GPU0 free of this run.
        gpu = next(value for value in (5, 7, 2, 3, 6, 1, 4, 0) if value in devices)

        def start(action, device):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if device is None else str(device),
                       OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='2')
            module = ('experiments.pamf_lora_main.dura_real_gain_r17_fit' if action == 'fit'
                      else 'experiments.pamf_lora_main.dura_real_gain_r17_screen')
            args = [sys.executable, '-u', '-m', module]
            if action != 'fit':
                args.append(action)
            args.append(str(runtime))
            with (root / f'{action}.log').open('x', encoding='utf-8') as log:
                proc = subprocess.Popen(args, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            write(state, dict(status=action.upper(), pid=proc.pid, gpu=device), exclusive=False)
            print('LORA_R17_GAIN_WORKER', action, proc.pid, device, flush=True)
            if proc.wait():
                raise RuntimeError(f'{action} failed; preserve partial artifacts')

        start('fit', gpu)
        start('predict', gpu)
        start('score', None)
        outcome = read(root / 'screen-score.json')
        write(state, dict(status='SOURCE_SCREEN_SCORED', gpu=gpu, promote=outcome['promote']), exclusive=False)
    except Exception as error:
        write(state, dict(status='FAILED', error=str(error)), exclusive=False)
        raise


def publish(repo, output, context, ssh_config):
    from experiments.pamf_lora_main.publish import release, release_commit
    source = read(context)
    rt = {key: source[key] for key in ('project', 'python', 'frozen_sources')}
    project = PurePosixPath(rt['project'])
    runs, releases = project / 'runs', project / 'releases'
    rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(repo),
              release_root=str(releases / f'{RUN}-release'), run_root=str(runs / RUN),
              external_releases=[str(runs / LORA_ACTION), str(runs / LORA_GAIN),
                                 str(releases / f'{LORA_GAIN}-release')])
    release(repo, output, rt, ssh_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    worker = commands.add_parser('dispatch')
    worker.add_argument('runtime', type=Path)
    pub = commands.add_parser('publish')
    for field in ('repo', 'output', 'context', 'ssh-config'):
        pub.add_argument('--' + field, type=Path, required=True)
    args = vars(parser.parse_args())
    action = args.pop('action')
    (dispatch if action == 'dispatch' else publish)(**args)
