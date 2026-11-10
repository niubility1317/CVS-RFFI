"""Use identical prediction/scoring matrix and both fixed reference controls."""
from pathlib import Path
from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_physics_branch_balanced import prepare as fixed
from experiments.pamf_physics_branch_balanced.plan import RUN as PREVIOUS
from .plan import RUN, candidate_rows, configuration


def prepare(rt):
    previous = Path(rt['project']) / 'runs' / PREVIOUS
    state = read(previous / 'pipeline_state.json')
    if state['status'] not in ('ANALYZED', 'PARTIAL_TECHNICAL_FAILURE') or any(v.get('status') == 'RUNNING' for v in state['jobs'].values()):
        raise ValueError('Fixed reference-budget owner must be terminal before adaptive execution')
    old = fixed.RUN, fixed.candidate_rows, fixed.configuration
    fixed.RUN, fixed.candidate_rows, fixed.configuration = RUN, candidate_rows, configuration
    try:
        jobs = fixed.prepare(rt)
        source_done = str(Path(rt['run_root']) / 'F' / candidate_rows()[0]['row_id'] / 'source' / 'complete.json')
        for job in jobs:
            if job['kind'] == 'source_diagnostic':
                job['nonblocking'] = True
            elif job['kind'] in ('predict', 'weak_predict'):
                job['dependencies'] = [source_done]
        write(Path(rt['run_root']) / 'jobs.json', jobs, exclusive=False)
        path = Path(rt['run_root']) / 'sources.json'
        sources = read(path)
        matched = read(previous / 'sources.json')['ptbr-refbudget-s2026092701']
        if read(Path(matched['scores']) / 'complete.json')['status'] != 'SCORED_COMPLETE':
            raise ValueError('Fixed reference-budget practical reference incomplete')
        if read(Path(matched['weak_scores']) / 'complete.json')['status'] != 'SCORED_WEAK_COMPLETE':
            raise ValueError('Fixed reference-budget weak reference incomplete')
        sources['ptbr-refbudget-s2026092701'] = dict(matched, reused=True)
        write(path, sources, exclusive=False)
        return jobs
    finally:
        fixed.RUN, fixed.candidate_rows, fixed.configuration = old
