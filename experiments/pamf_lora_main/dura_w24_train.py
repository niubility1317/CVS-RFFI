"""Train W24 DURA from raw, unlabeled source IQ in one 40,000-step loop."""

import argparse
import os
from pathlib import Path
import subprocess

import torch

from experiments.cric_dura_comparison.bootstrap import numerical_policy
from experiments.cric_dura_comparison.io import read, tensor, write
from .channel_blind_source import FitRecipe, fit, partition
from .dura_w24_model import DuraW24, WideTracker, calibrate, objective
from .full_packet_views import source_contract, source_role
from .posterior_transport import condition_waveforms
from .publish import release, release_commit
from .refine_runner import select_gpu
from .weak_guard_target import DATA


RUN = '20261125-phase1-lora-dura-w24-train-s2026112991-r01'
RECIPE = FitRecipe('w24', 40_000, DuraW24, condition_waveforms, None,
                   coarse_first=True, batch=16, fit_per_rx=4000,
                   validation_steps=(1800, 6000, 12000, 20000, 30000), finalize_calibration=calibrate)


def prepare(rt, device, *, smoke, fit_per_rx, bank_per_rx):
    source = Path(rt['source_root'])
    contract = source_contract(source)
    iq, rx, _, _ = source_role(source, contract, 'U_s')
    roles = partition(rx, fit=5, cal=4, held=3) if smoke else partition(rx, fit=fit_per_rx)
    return tuple(tensor(iq[index].view('float32'), device).view(torch.complex64) for index in roles), contract


def dispatch(rt, runtime):
    root = Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    state = root / 'state.json'
    (root / 'logs').mkdir()

    def launch(action, gpu):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OPENBLAS_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', OMP_NUM_THREADS='2')
        cmd = [rt['python'], '-u', '-m', __spec__.name, action, str(runtime)]
        with (root / 'logs' / f'{action}.log').open('x') as log:
            job = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        print('W24_DURA_LAUNCHED', action, job.pid, gpu, flush=True)
        return job

    try:
        gpu = select_gpu()
        write(state, dict(status='SOURCE_SMOKE', gpu=gpu))
        if launch('fit_smoke', gpu).wait():
            raise RuntimeError('W24 real U source fit smoke failed')
        gpu = select_gpu()
        write(state, dict(status='SOURCE_TRAINING', gpu=gpu, steps=RECIPE.steps), exclusive=False)
        if launch('fit', gpu).wait():
            raise RuntimeError('W24 40,000-step fit failed')
        terminal = read(root / 'w24-complete.json')
        if (terminal['status'], terminal['steps']) != ('SOURCE_FROZEN', RECIPE.steps) or not (root/'w24-model.pt').is_file():
            raise ValueError('Expected the terminal W24 source weights')
        write(state, dict(status='SOURCE_FROZEN', steps=RECIPE.steps), exclusive=False)
    except Exception as error:
        write(state, dict(status='FAILED', error=str(error)), exclusive=False)
        raise


def publish(repo, output, context, ssh_config):
    cfg = read(context)
    project = cfg['project']
    rt = {key: cfg[key] for key in ('project', 'python', 'frozen_sources')}
    rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(repo),
              release_root=f'{project}/releases/{RUN}-release', run_root=f'{project}/runs/{RUN}',
              source_root=f'{project}/releases/{DATA}-release/data/source')
    release(repo, output, rt, ssh_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    publisher = commands.add_parser('publish')
    for key in ('repo', 'output', 'context', 'ssh-config'):
        publisher.add_argument(f'--{key}', required=True, type=Path)
    for action in ('dispatch', 'fit_smoke', 'fit'):
        commands.add_parser(action).add_argument('runtime', type=Path)
    args = vars(parser.parse_args())
    command = args.pop('command')
    if command == 'publish':
        publish(**args)
    else:
        rt = read(args['runtime'])
        if rt['run_id'] != RUN:
            raise ValueError('The W24 fit requires its own immutable source-only run')
        if command == 'dispatch':
            dispatch(rt, args['runtime'])
        else:
            numerical_policy()
            fit(rt, smoke=command=='fit_smoke', recipe=RECIPE, calculate=objective,
                receiver_batches=True, prepare=prepare, make_tracker=WideTracker)
