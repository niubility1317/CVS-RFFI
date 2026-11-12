"""Staged Hybrid confidence-only training, direct prediction and truth-last scoring."""
import argparse
from contextlib import contextmanager
from pathlib import Path

from experiments.cric_dura_comparison.io import read, write
from . import hybrid_confidence_plan as plan, target_tuning_runner as base
from .hybrid_confidence import HybridConfidence
from .practical_gate import installed as practical_scope


@contextmanager
def installed(c, holder):
    from experiments.pamf_paper_completion.model import installed as paper_scope

    @contextmanager
    def adapted(cfg, state, enabled, objective_factory):
        with paper_scope(cfg, state, ssl_enabled=enabled, objective_factory=objective_factory,
                         validate=plan.validate):
            yield

    with practical_scope(c, holder, base_scope=adapted, objective_factory=HybridConfidence):
        yield


def prepare(rt):
    old = base.plan
    base.plan = plan
    try:
        return base.prepare(rt)
    finally:
        base.plan = old


def stage(rt):
    from experiments.practical_full_2x2.cache import atomic
    root = Path(rt['run_root'])
    root.mkdir(parents=True, exist_ok=False)
    (root / 'logs').mkdir()
    jobs = prepare(rt)
    smoke_output = root / 'smoke.json'
    for job in jobs:
        if job['kind'] == 'train':
            job['dependencies'].append(str(smoke_output))
    jobs.insert(0, dict(id='smoke', kind='smoke', config=rt['runtime_file'], completion=str(smoke_output),
                        successful_statuses=['PASS'], dependencies=[]))
    write(root / 'jobs.json', jobs, exclusive=False)
    atomic(root / 'pipeline_state.json', dict(status='STAGED', pid=None, cwd=rt['release_root'], argv=[],
           code_commit=rt['code_commit'], owner=rt['launch_owner'], admission_mib=7258,
           jobs={job['id']: dict(status='QUEUED') for job in jobs}, failed_jobs=[]))


def check_hybrid(state):
    from .pair_runner import check_single_rx
    details = check_single_rx(state)
    route = state['U_confidence_only']
    if not route['used'] or route['used'] != state['accepted'] or route['branch_votes_used'] or route['prototype_used']:
        raise ValueError('All confidence-accepted U must reach Hybrid without branch votes or prototypes')
    return dict(details, U_confidence_only=route)


def smoke(rt):
    from .runner import smoke as smoke_rows
    checks = []
    for row in (r for r in plan.candidate_rows() if r['model_seed'] == plan.SEEDS[0]):
        checks.extend(smoke_rows(rt, rows=lambda: [row], configure=plan.configuration, check_parent=plan.parent,
                                scope=installed, check_u=check_hybrid, graph_check=plan.check_graph,
                                segment_output=True, write_result=False))
    write(Path(rt['run_root']) / 'smoke.json', dict(status='PASS', checks=checks, real_checkpoint=True,
          optimizer_updates=0, query_reads=0, U_truth_reads=0))


def worker(c, kind):
    if kind == 'score':
        from experiments.pamf_paper_completion.evaluation import score
        return score(c)
    from experiments.pamf_repl_completion.runner import activate
    activate(c)
    if kind == 'smoke':
        return smoke(c)
    if kind == 'train':
        from .source import train_trajectory
        return train_trajectory(c, plan.parent(c), installed, ssl=True, physical_L=False)
    if kind == 'predict':
        from .development_evaluation import predict
        return predict(c)
    raise ValueError('Only source training, frozen prediction and independent scoring are allowed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['worker', 'stage'])
    parser.add_argument('--kind')
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = read(args.config)
    stage(config) if args.action == 'stage' else worker(config, args.kind)
