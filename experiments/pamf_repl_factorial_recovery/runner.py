"""New owner; preserve prior frozen E50 and rerun only unfinished source rows."""
import argparse
from functools import partial
from pathlib import Path
from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_repl_component_guard import plan as base_plan, runner as base
from experiments.pamf_repl_step_guard import runner as scored
from experiments.pamf_repl_attribution_factorial.objective import installed
from . import plan


def prepare(rt):
    old_root = Path(rt['project']) / 'runs' / plan.PREVIOUS_RUN
    state = read(old_root / 'pipeline_state.json')
    if state['status'] != 'PARTIAL_TECHNICAL_FAILURE':
        raise ValueError('Previous owner must be stopped after the technical failure')
    old_sources = read(old_root / 'sources.json')
    jobs = scored.prepare(rt)
    root = Path(rt['run_root'])
    sources = read(root / 'sources.json')
    reused = set()
    for row in plan.candidate_rows():
        rid = row['row_id']
        status = state['jobs']['train__' + rid]['status']
        if status == 'COMPLETE':
            folder = Path(old_sources[rid]['source_output'])
            done = read(folder / 'complete.json')
            cfg = done['config']
            if (done['status'] != 'SOURCE_FROZEN' or done['epoch'] != 50
                    or done['target_access'] or done['U_truth_read']
                    or cfg['row_id'] != rid or cfg['run_id'] != plan.PREVIOUS_RUN
                    or cfg['tuning_recipe'] != plan.RECIPES[row['recipe']]):
                raise ValueError('Reused E50 is not the exact source-only row')
            if not (folder / 'final_ssdg.pth').is_file():
                raise FileNotFoundError('Frozen checkpoint missing')
            for epoch in (30, 50):
                if not (folder / f'segment{epoch}' / 'U_pseudo_observations.jsonl.gz').is_file():
                    raise FileNotFoundError('Frozen blind observation missing')
            reused.add(rid)
            sources[rid]['source_output'] = str(folder)
            for kind in ('predict', 'u_score'):
                job = next(j for j in jobs if j['id'] == kind + '__' + rid)
                cfg = read(job['config']); cfg['source_output'] = str(folder)
                write(job['config'], cfg, exclusive=False)
                job['dependencies'] = [str(folder / 'complete.json')]
        elif status not in ('FAILED', 'QUEUED'):
            raise ValueError('Unexpected prior source state: ' + rid + ' ' + status)
    if len(reused) != 2:
        raise ValueError('Expected exactly two already frozen rows')
    jobs = [j for j in jobs if not (j['kind'] == 'train' and j['id'].split('__', 1)[1] in reused)]
    write(root / 'sources.json', sources, exclusive=False)
    write(root / 'jobs.json', jobs, exclusive=False)
    return jobs


def worker(c, kind):
    if kind == 'score' and c.get('diagnostic_type') == 'U_truth_after_freeze':
        from experiments.pamf_repl_step_guard.u_scoring import score
        return score(c)
    return base.worker(c, kind)


def dispatch(rt):
    from experiments.pamf_branch_training import runner as queue, prepare as preparation
    from experiments.pamf_physics_branch.runner import admits as physical_admits
    from experiments.pamf_physics_route.runner import launch
    def admits(gpu, active, mib):
        if any(item['gpu'] == gpu for _, item in active.values()):
            return False
        return physical_admits(gpu, active, max(mib, 12000), max_total=1)
    old = preparation.prepare, queue.admits, queue.launch, queue.analyze
    preparation.prepare, queue.admits = prepare, admits
    queue.launch = partial(launch, module='experiments.pamf_repl_factorial_recovery.runner')
    queue.analyze = base.analyze
    try:
        return queue.dispatch(dict(rt, prioritize_prediction=True))
    finally:
        preparation.prepare, queue.admits, queue.launch, queue.analyze = old


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['dispatch', 'worker'])
    parser.add_argument('--kind'); parser.add_argument('--config', required=True)
    args = parser.parse_args(); c = read(args.config)
    with plan.scoped():
        previous = base.installed, scored.plan
        base.installed, scored.plan = installed, base_plan
        try:
            dispatch(c) if args.action == 'dispatch' else worker(c, args.kind)
        finally:
            base.installed, scored.plan = previous
