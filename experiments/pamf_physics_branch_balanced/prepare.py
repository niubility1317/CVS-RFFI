"""Reuse the fixed source and truth-last matrix with a distinct run owner."""
from experiments.pamf_physics_branch import prepare as physical
from experiments.pamf_physics_branch.plan import RUN as PREVIOUS
from experiments.cric_dura_comparison.io import read, write
from pathlib import Path
from .plan import RUN, candidate_rows, configuration


def prepare(rt):
    previous = Path(rt['project']) / 'runs' / PREVIOUS
    state = read(previous / 'pipeline_state.json')
    if state['status'] != 'ANALYZED' or any(v.get('status') == 'RUNNING' for v in state['jobs'].values()):
        raise ValueError('The matched PTBR owner must finish before reference-budget execution')
    old = physical.RUN, physical.candidate_rows, physical.configuration
    physical.RUN, physical.candidate_rows, physical.configuration = RUN, candidate_rows, configuration
    try:
        jobs = physical.prepare(rt)
        source_path = Path(rt['run_root']) / 'sources.json'
        sources = read(source_path)
        matched = read(previous / 'sources.json')['ptbr-beta05-s2026092701']
        if read(Path(matched['scores']) / 'complete.json')['status'] != 'SCORED_COMPLETE':
            raise ValueError('Matched beta=0.5 practical reference incomplete')
        if read(Path(matched['weak_scores']) / 'complete.json')['status'] != 'SCORED_WEAK_COMPLETE':
            raise ValueError('Matched beta=0.5 weak reference incomplete')
        sources['ptbr-beta05-s2026092701'] = dict(matched, reused=True)
        write(source_path, sources, exclusive=False)
        return jobs
    finally:
        physical.RUN, physical.candidate_rows, physical.configuration = old
