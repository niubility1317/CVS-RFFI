"""Hu-only Phase1 readouts and non-invasive admission after the Phase2 owner."""
import os
from pathlib import Path
import time

from .group1 import source_scope
from .io import atomic, read, require


def alive(pid):
    try:
        return Path('/proc', str(pid), 'stat').read_text().split(') ', 1)[1][0] != 'Z'
    except (FileNotFoundError, ProcessLookupError):
        return False


def wait_for_previous_owner(rt, root):
    """No capacity race: never overlap the existing independent Phase2 owner."""
    while True:
        previous = read(rt['wait_for_phase2_state'])
        active = previous.get('active', {})
        finished = previous.get('status') in ('COMPLETE', 'PARTIAL_TECHNICAL_FAILURE')
        idle = not alive(previous['controller_pid']) and not any(alive(v['pid']) for v in active.values())
        if finished and idle:
            return
        atomic(root/'pipeline_state.json', dict(status='WAITING_FOR_PHASE2_OWNER',
            controller_pid=os.getpid(), previous_status=previous.get('status'),
            previous_controller_pid=previous['controller_pid'], updated_at=time.time()))
        if (root/'USER_STOP.json').exists():
            raise RuntimeError('User stopped queued Hu run; no workers launched')
        time.sleep(rt.get('poll_seconds', 30))


def phase1_jobs(rt):
    phase1 = read(rt['hu_phase1_runtime'])
    h = read(phase1['context'])['handles']['F0']
    require(h['classes'] == rt['old_classes'], 'Hu Phase1 class ordering differs')
    scoped = source_scope(rt)
    require(read(h['source_contract']) == read(scoped['source_contract']), 'Hu source contracts differ')
    require(rt['method_profiles'] == ['feature_separation_hu2024'], 'Hu-only release expected')
    tests = ('cross_rx', 'cross_both')
    root = Path(rt['run_root']); jobs = []
    for seed in rt['model_seeds']:
        row = f'feature_separation_hu2024-s{seed}'
        base = root/'phase1'/'readouts'/row
        pred = dict(run_id=rt['run_id'], row_id=row, profile='feature_separation_hu2024',
            model_seed=seed, source_runtime=scoped, classes=h['classes'], source_days=[1,2,3],
            readouts=['raw'], tests={t: {'views_root': h['tests'][t]['views_root']} for t in tests},
            output_root=str(base/'predictions'))
        score = dict(run_id=rt['run_id'], row_id=row, model_seed=seed, classes=h['classes'],
            source_days=[1,2,3], readouts=['raw'], tests={t:h['tests'][t] for t in tests},
            prediction_output=pred['output_root'], output_root=str(base/'scores'))
        name = 'phase1-predict-'+row
        jobs.append(dict(id=name, kind='phase1_predict', profile='feature_separation_hu2024',
            seed=seed, config=pred, gpu=True, dependencies=['source-'+row], priority=2,
            completion=str(base/'predictions'/'complete.json'), status='ALL_PREDICTIONS_COMPLETE'))
        jobs.append(dict(id='phase1-score-'+row, kind='phase1_score', profile='feature_separation_hu2024',
            seed=seed, config=score, gpu=False, dependencies=[name], priority=0,
            completion=str(base/'scores'/'complete.json'), status='SCORED_COMPLETE'))
    return jobs
