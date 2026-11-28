"""Source-only paired diagnostic of original versus packet-normalized R17 path."""
import argparse
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys

import torch
from torch import nn

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main.dura_real_gain_fit import (LORA_ACTION, LORA_GAIN, WISIG_CACHE,
    WISIG_PACKET, WISIG_PARENT)
from experiments.pamf_lora_main.shared_dura_realgain_run import REFERENCE, dispatch, score
from experiments.pamf_lora_main import shared_twoprior_run as shared

RUN = '20261128-diagnostic-shared-r17-scale-s2026112953-r01'
SEED, STEPS = 2026112953, 0
PARENT = '20261128-phase1-dura-realgain-crossdata-s2026112845-r02'


class NormalizedR17(nn.Module):
    def __init__(self, width):
        super().__init__()
        if width not in (32, 1024):
            raise ValueError('Unknown packet width')

    def forward(self, received, real_gain, r17, action):
        return r17


def worker(runtime, dataset):
    rt = read(runtime)
    folder = Path(rt['run_root']) / dataset
    shared.SharedTwoPriorGain = NormalizedR17
    shared.SEED, shared.STEPS = SEED, STEPS
    proof = dict(status='SOURCE_FROZEN', dataset=dataset, seed=SEED, updates=0,
                 F_reads=0, TX_reads=0, V_reads=0, target_reads=0, parameters=0,
                 rule='R17(received/RMS)*RMS; compare original R17(received); no training')
    with (folder / 'gain.pt').open('xb') as stream:
        torch.save(dict(weights=NormalizedR17(shared.WIDTH[dataset]).state_dict(), proof=proof), stream)
    write(folder / 'fit.json', proof)
    shared.predict(rt, dataset, candidate_name='normalized_r17')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    result = subprocess.run([sys.executable, '-u', '-m', __spec__.name, 'score', str(runtime), dataset],
                            env=env, stdin=subprocess.DEVNULL, check=False)
    if result.returncode:
        raise RuntimeError('Independent source diagnostic scoring failed')


def diagnostic_score(rt, dataset):
    score(rt, dataset, candidate_name='normalized_r17')


def publish(repo, output, context, ssh_config):
    from experiments.pamf_lora_main.publish import release, release_commit
    source = read(context)
    rt = {k: source[k] for k in ('project', 'python', 'frozen_sources')}
    project = PurePosixPath(rt['project'])
    runs, releases = project / 'runs', project / 'releases'
    rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(repo),
              release_root=str(releases / f'{RUN}-release'), run_root=str(runs / RUN),
              lora_frozen_root=str(runs / PARENT), allowed_gpus=[7, 1, 3, 6],
              external_releases=[str(runs / n) for n in (LORA_ACTION, LORA_GAIN, WISIG_CACHE,
                  WISIG_PACKET, WISIG_PARENT, PARENT,
                  '20261128-diagnostic-lora-r69-full-w24-s2026112895-r01')] + [str(releases / REFERENCE)])
    release(repo, output, rt, ssh_config, archive_extras=('tools/r17_activation_factorization.py',))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    for phase in ('dispatch', 'worker', 'score'):
        child = commands.add_parser(phase)
        child.add_argument('runtime', type=Path)
        if phase != 'dispatch':
            child.add_argument('dataset', choices=('lora', 'wisig'))
    pub = commands.add_parser('publish')
    for field in ('repo', 'output', 'context', 'ssh-config'):
        pub.add_argument('--' + field, type=Path, required=True)
    args = vars(parser.parse_args())
    action = args.pop('action')
    if action == 'dispatch':
        dispatch(args['runtime'], worker_module=__spec__.name)
    elif action == 'score':
        diagnostic_score(read(args['runtime']), args['dataset'])
    elif action == 'worker':
        worker(**args)
    else:
        publish(**args)
