"""One-seed POSTER v2 A/B pilot with a separate truth-last scorer."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

from experiments.pamf_phase2_practical.io import read, require, write

PROFILE = 'poster_ft_paperlast_v2'


def run_stage(rt, stage):
    root = Path(rt['run_root'])
    seed = int(rt['pilot_seed'])
    if stage == 'source':
        from experiments.pamf_phase2_practical.group1 import source_scope
        from experiments.pamf_phase2_practical.group1_source import train
        return train(source_scope(rt), PROFILE, seed, root / 'source' / f'{PROFILE}-s{seed}')
    if stage == 'predict':
        from experiments.pamf_phase2_practical.group1_target import predict
        return predict(rt, dict(profile=PROFILE, seed=seed, capsule=rt['pilot_capsule'],
                                scope='pilot_main', output=str(root / 'predictions')))
    if stage == 'score':
        from experiments.pamf_phase2_practical.episodic_score import run
        return run(rt, dict(prediction=str(root / 'predictions'), output=str(root / 'scores')))
    raise ValueError('Unknown pilot stage')


def dispatch(rt, runtime_path):
    root = Path(rt['run_root'])
    require(rt.get('reuse_source_augmentation') is True, 'Pilot requires read-only source cache')
    require(rt.get('pilot_capsule') == 'capsules/P01/MIX01/1-1', 'Unexpected pilot capsule')
    require(int(rt.get('pilot_seed', 0)) == 2026092701, 'Unexpected pilot seed')
    require(not rt.get('clean_root'), 'Pilot uses satellite-received data')
    root.mkdir(parents=True, exist_ok=False)
    write(root / 'owner.json', dict(pid=os.getpid(), profile=PROFILE, runtime=str(runtime_path)))
    artifacts = {
        'source': (root / 'source' / f"{PROFILE}-s{rt['pilot_seed']}" / 'group1_complete.json', 'GROUP1_SOURCE_READY'),
        'predict': (root / 'predictions' / 'complete.json', 'PREDICTIONS_COMPLETE'),
        'score': (root / 'scores' / 'complete.json', 'SCORED_COMPLETE'),
    }
    for stage in ('source', 'predict', 'score'):
        env = dict(os.environ)
        if stage == 'score':
            env['CUDA_VISIBLE_DEVICES'] = ''
        command = [sys.executable, '-u', '-m', 'experiments.poster_paper_repro.phase2_pilot',
                   '--runtime', str(runtime_path), '--stage', stage]
        print('PILOT_STAGE_START', stage, flush=True)
        subprocess.run(command, cwd=rt['release_root'], env=env, check=True)
        artifact, expected = artifacts[stage]
        require(artifact.is_file() and read(artifact).get('status') == expected,
                f'{stage} did not produce its expected completion')
        print('PILOT_STAGE_COMPLETE', stage, flush=True)
    write(root / 'complete.json', dict(status='SCORED_COMPLETE', profile=PROFILE, seed=rt['pilot_seed'],
        capsule=rt['pilot_capsule'], truth_last=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', required=True)
    parser.add_argument('--stage', choices=('source', 'predict', 'score'))
    args = parser.parse_args()
    rt = read(args.runtime)
    if args.stage:
        run_stage(rt, args.stage)
    else:
        dispatch(rt, args.runtime)


if __name__ == '__main__':
    main()
