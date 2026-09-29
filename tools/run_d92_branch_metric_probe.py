"""CPU-only within-class metric diagnostic from existing support-only feature caches."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import sys
import threading
import time

from run_d92_branch_support_probe import read, write, launch


def validate_spec(spec):
    rows = spec['rows']; root = Path(spec['execution']['remote_run_root']).resolve()
    if (spec['permissions']['query_use'] != 'none; query IQ/labels/truth/scores never read'
            or spec['probe']['query_access'] is not False or spec['probe']['reuse_support_cache'] is not True
            or spec['execution']['cpu_lanes'] != 4 or spec['execution']['blas_threads_per_lane'] != 2
            or spec['execution']['export_device'] is not None):
        raise ValueError('CPU support-only contract mismatch')
    if len(rows) != 8 or len({r['row_id'] for r in rows}) != 8:
        raise ValueError('Complete eight-row matrix required')
    pairs = set()
    for row in rows:
        out = Path(row['output_root']).resolve(); co = spec['probe']['cohorts'][row['cohort']]
        if out.parent != root or out.name != row['row_id']:
            raise ValueError('Row output escaped run root')
        if row['data_overrides'] != dict(capsule=co['capsule'], capsule_id=co['capsule_id'], expected_split_count=co['expected_split_count']):
            raise ValueError('Cohort binding mismatch')
        cache = Path(row['support_features']).resolve()
        if cache == root or root in cache.parents or cache == out or out in cache.parents:
            raise ValueError('Existing cache must be outside new outputs')
        digest = row['expected_checkpoint_sha256']
        if len(digest) != 64 or any(v not in '0123456789abcdef' for v in digest):
            raise ValueError('Invalid checkpoint binding')
        pairs.add((row['cohort'], row['seeds']['model']))
    if pairs != {(c, s) for c in ('rx3', 'rx1') for s in range(2026092701, 2026092705)}:
        raise ValueError('Model/cohort coverage mismatch')


def command(spec, row):
    co = spec['probe']['cohorts'][row['cohort']]
    return [sys.executable, '-u', str(Path(spec['code']['cwd'])/'tools/evaluate_d92_branch_metric_probe.py'),
        '--support-features', row['support_features'], '--capsule', co['capsule'],
        '--output', str(Path(row['output_root'])/'probe'), '--config', co['evaluation_config'],
        '--expected-capsule-id', co['capsule_id'], '--expected-checkpoint-sha256', row['expected_checkpoint_sha256'],
        '--expected-model-seed', str(row['seeds']['model'])]


def verify_marker(path, spec, row):
    marker = read(path); co = spec['probe']['cohorts'][row['cohort']]
    if (marker['status'] != 'SUPPORT_PROBE_COMPLETE' or marker['capsule_id'] != co['capsule_id']
            or marker['checkpoint_sha256'] != row['expected_checkpoint_sha256']
            or marker['episodes'] != co['expected_split_count']
            or marker['k1_episodes'] != co['expected_split_count']//4
            or marker['oof_episodes'] != 3*co['expected_split_count']//4
            or marker['proxy_anchor_count'] != 35*co['expected_split_count']//4
            or marker['query_rows_used'] != 0 or marker['source_rows_used'] != 0):
        raise ValueError('Incomplete or incorrectly bound support diagnostic')
    return marker


def run(spec, commit, launch_fn=launch):
    validate_spec(spec)
    root = Path(spec['execution']['remote_run_root']); root.mkdir(parents=True, exist_ok=False)
    rows = spec['rows']; state = {r['row_id']: dict(status='PENDING') for r in rows}; lock = threading.Lock()
    write(root/'startup.json', dict(pid=os.getpid(), argv=sys.argv, cwd=os.getcwd(), python=sys.executable,
        commit=commit, spec=spec, started=time.time(), query_access=False, source_sample_access=False,
        checkpoint_loaded=False, reused_support_cache=True, gpu_use=False))
    def update(row, status, **values):
        with lock:
            state[row['row_id']] = dict(status=status, updated=time.time(), **values)
            write(root/'state.json', state)
    def work(row):
        out = Path(row['output_root'])
        try:
            out.mkdir(exist_ok=False); update(row, 'PROBING_SUPPORT')
            launch_fn(command(spec, row), out/'probe.log', Path(spec['code']['cwd']), 'probe')
            marker = verify_marker(out/'probe/probe_complete.json', spec, row)
            update(row, 'SUPPORT_PROBE_COMPLETE', episodes=marker['episodes'],
                k1_episodes=marker['k1_episodes'], oof_episodes=marker['oof_episodes'],
                proxy_anchor_count=marker['proxy_anchor_count'],
                factorization_count=marker['factorization_count'])
        except Exception as exc:
            update(row, 'FAILED', error=str(exc))
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(work, rows))
    success = all(v['status'] == 'SUPPORT_PROBE_COMPLETE' for v in state.values())
    write(root/'complete.json', dict(status='SUPPORT_PROBE_COMPLETE' if success else 'FAILED', commit=commit,
        model_rows=len(rows), completed_rows=sum(v['status']=='SUPPORT_PROBE_COMPLETE' for v in state.values()),
        episodes=sum(v.get('episodes', 0) for v in state.values()), query_access=False, source_sample_access=False,
        k1_episodes=sum(v.get('k1_episodes', 0) for v in state.values()),
        oof_episodes=sum(v.get('oof_episodes', 0) for v in state.values()),
        proxy_anchor_count=sum(v.get('proxy_anchor_count', 0) for v in state.values()),
        factorization_count=sum(v.get('factorization_count', 0) for v in state.values()),
        optimizer_steps=0, query_performance_claim=False, checkpoint_loaded=False, gpu_use=False, finished=time.time()))
    if not success:
        raise RuntimeError('Support diagnostic lane failed; artifacts preserved; no automatic retry')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--spec', type=Path, required=True); p.add_argument('--commit', required=True)
    a = p.parse_args(); run(read(a.spec), a.commit)
