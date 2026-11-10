"""One owner, source-only route-stability training, practical-only testing."""
import argparse
import csv
from functools import partial
import os
from pathlib import Path
import subprocess
import time
from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_branch_training import runner as base
from experiments.pamf_physics_branch import runner as physical
from experiments.pamf_physics_branch.route_training_scope import installed
from .plan import candidate_rows, configuration
from .prepare import prepare
from .source import checked_parent


def worker(c, kind):
    if kind not in ('smoke', 'train'):
        if kind not in ('source_diagnostic', 'predict', 'score'):
            raise ValueError('Only practical evaluation stages are permitted')
        return physical.worker(c, kind)
    from experiments.pamf_physics_branch import source, smoke
    old = source.checked_parent, source.installed, smoke.checked_parent, smoke.candidate_rows, smoke.configuration, smoke.installed
    source.checked_parent, source.installed = checked_parent, installed
    smoke.checked_parent = checked_parent
    smoke.candidate_rows = partial(candidate_rows, c['run_id'])
    smoke.configuration = partial(configuration, run_id=c['run_id'])
    smoke.installed = installed
    try:
        from experiments.pamf_repl_completion.runner import activate
        activate(c)
        if kind == 'train':
            return source.train(c)
        result = smoke.smoke(c)
        for check in read(Path(c['run_root']) / 'smoke.json')['checks']:
            evidence = check['U']['reference_routing']
            if evidence['calls'] != 2 or evidence['selected_samples'] != 192:
                raise ValueError('Actual source smoke did not execute both L and U routing')
        return result
    finally:
        (source.checked_parent, source.installed, smoke.checked_parent, smoke.candidate_rows,
         smoke.configuration, smoke.installed) = old


def launch(rt, job, gpu):
    path = Path(rt['run_root']) / 'logs' / (job['id'] + '.log')
    cmd = [rt['python'], '-u', '-m', 'experiments.pamf_physics_route.runner',
           'worker', '--kind', job['kind'], '--config', job['config']]
    env = dict(os.environ, PYTHONPATH=rt['release_root'], CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONUNBUFFERED='1')
    with path.open('x') as log:
        process = subprocess.Popen(cmd, cwd=rt['release_root'], env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return process, dict(status='RUNNING', pid=process.pid, gpu=gpu, cwd=rt['release_root'],
                         argv=cmd, log=str(path), started_at=time.time())


def analyze(rt):
    root = Path(rt['run_root'])
    records = []
    for rid, source in read(root / 'sources.json').items():
        folder = Path(source['scores'])
        data = read(folder / 'scores.json')
        if read(folder / 'complete.json')['status'] != 'SCORED_COMPLETE' or data['truth_last'] is not True:
            raise ValueError('Complete independently scored practical rows required')
        for r in data['results']:
            if r['dimension'] == 'overall':
                records.append(dict(row=rid, test=r['test'], readout=r['readout'], view=r['view'],
                                    count=r['count'], accuracy=r['accuracy'], macro_f1=r['macro_f1']))
    out = root / 'analysis'
    out.mkdir(exist_ok=False)
    with (out / 'overall.csv').open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    write(out / 'complete.json', dict(status='ANALYZED', rows=len(read(root / 'sources.json')),
          test_scope='clean_and_six_practical', target_feedback_forbidden=True, results=len(records)))


def dispatch(rt):
    from experiments.pamf_branch_training import prepare as old_prepare
    old = old_prepare.prepare, base.admits, base.launch, base.analyze
    old_prepare.prepare, base.admits, base.launch, base.analyze = prepare, physical.admits, launch, analyze
    try:
        base.dispatch(rt)
    finally:
        old_prepare.prepare, base.admits, base.launch, base.analyze = old


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['dispatch', 'worker'])
    p.add_argument('--kind')
    p.add_argument('--config', required=True)
    a = p.parse_args()
    dispatch(read(a.config)) if a.action == 'dispatch' else worker(read(a.config), a.kind)
