"""Re-evaluate all fixed seeds with repaired numerics, preserving the original run."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from cvs_native_artifacts import verify_source
from cvs_matched_pipeline import read, write

ROOT = Path(__file__).resolve().parents[1]


def verify_reuse(spec):
    contract = read(spec['source_contract'])
    capsule = read(Path(spec['phase2_capsule']) / 'manifest.json')
    for row in spec['rows']:
        source = Path(row['source_root'])
        original = Path(row['reuse_root'])
        verify_source(source, contract, row['seeds']['model'])
        ground = read(original / 'ground/manifest.json')
        features = read(original / 'received_features/features_complete.json')
        if ground['provenance_status'] != 'CURRENT_FINAL_SCRATCH_EXACT_SOURCE_L' or ground['target_access'] is not False:
            raise ValueError('Unverified inherited ground')
        if ground['checkpoint_sha256'] != hashlib.sha256((source / 'final_ssdg.pth').read_bytes()).hexdigest():
            raise ValueError('Ground checkpoint mismatch')
        if features['status'] != 'FROZEN_FEATURES_COMPLETE' or features['capsule_id'] != capsule['capsule_id'] or features['query_used_for_fitting'] is not False:
            raise ValueError('Unverified frozen features')
        if Path(row['output_root']).exists():
            raise FileExistsError('Recovery row already exists; reconcile before retry')


def dispatch(spec, specpath, commit):
    root = Path(spec['execution']['remote_run_root'])
    state = {}; jobs = {}
    for row in spec['rows']:
        original = Path(row['reuse_root'])
        argv = [sys.executable, str(ROOT / 'tools/cvs_d92_matched.py'), 'predict',
                '--features', str(original / 'received_features/received_features.npz'),
                '--ground', str(original / 'ground'), '--capsule', spec['phase2_capsule'],
                '--output', row['output_root'], '--seed', str(row['seeds']['model']), '--resume-from', str(original)]
        with Path(row['log_path']).open('x', encoding='utf-8') as log:
            log.write(json.dumps(dict(argv=argv, cwd=str(ROOT), commit=commit, row=row)) + '\n'); log.flush()
            proc = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        jobs[row['row_id']] = proc
        state[row['row_id']] = dict(status='RUNNING', pid=proc.pid, argv=argv, log=row['log_path'])
        write(root / 'state.json', state)
    while jobs:
        for rowid, proc in list(jobs.items()):
            code = proc.poll()
            if code is not None:
                row = next(r for r in spec['rows'] if r['row_id'] == rowid)
                marker = Path(row['output_root']) / 'predictions_complete.json'
                complete = read(marker) if marker.exists() else {}
                ok = code == 0 and complete.get('status') == 'PREDICTIONS_COMPLETE' and complete.get('split_count') == 2100 and complete.get('predictions') == 2121
                state[rowid].update(status='PREDICTIONS_COMPLETE' if ok else 'TECHNICAL_FAILURE', exit_code=code)
                del jobs[rowid]; write(root / 'state.json', state)
        if jobs:
            time.sleep(15)
    if any(row['status'] != 'PREDICTIONS_COMPLETE' for row in state.values()):
        raise RuntimeError('Incomplete recovery matrix; preserve outputs and do not score')
    argv = [sys.executable, spec['phase2_scorer'], '--run-root', str(root), '--truth', spec['phase2_truth']]
    with (root / 'phase2_score.log').open('x', encoding='utf-8') as log:
        log.write(json.dumps(dict(argv=argv, cwd=str(ROOT))) + '\n'); log.flush()
        subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    if read(root / 'scored_results.json')['status'] != 'SCORED':
        raise ValueError('Missing scoring completion')
    write(root / 'completion.json', dict(status='SCORED', rows=len(state), commit=commit, spec=str(specpath)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['launch', 'dispatch'])
    p.add_argument('--spec', required=True, type=Path)
    p.add_argument('--commit', required=True)
    a = p.parse_args(); spec = read(a.spec); root = Path(spec['execution']['remote_run_root'])
    if a.action == 'dispatch':
        dispatch(spec, a.spec, a.commit)
        return
    if root.exists():
        raise FileExistsError('Recovery run already exists; reconcile before retry')
    verify_reuse(spec)
    root.mkdir(parents=True, exist_ok=False)
    write(root / 'experiment.json', spec)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1')
    argv = [sys.executable, str(Path(__file__).resolve()), 'dispatch', '--spec', str(a.spec.resolve()), '--commit', a.commit]
    with (root / 'dispatcher.log').open('x', encoding='utf-8') as log:
        proc = subprocess.Popen(argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write(root / 'launch.json', dict(pid=proc.pid, argv=argv, cwd=str(ROOT), commit=a.commit, owner=spec['execution']['launch_owner']))
    print(json.dumps(dict(pid=proc.pid, run_root=str(root))))


if __name__ == '__main__':
    main()
