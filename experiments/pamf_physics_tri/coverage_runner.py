"""Full-source coverage study with per-row frozen seven-view testing."""
import argparse
from contextlib import contextmanager
from functools import partial
from pathlib import Path

from experiments.cric_dura_comparison.io import read, write
from . import coverage_plan as plan, target_tuning_runner as base
from .coverage_objective import CoveragePrototype
from .practical_gate import installed as practical_scope


@contextmanager
def installed(c, holder):
    from experiments.pamf_paper_completion.model import installed as paper_scope

    @contextmanager
    def adapted(cfg, state, enabled, objective_factory):
        with paper_scope(cfg, state, ssl_enabled=enabled, objective_factory=objective_factory,
                         validate=plan.validate):
            yield

    with practical_scope(c, holder, base_scope=adapted, objective_factory=CoveragePrototype):
        yield


def prepare(rt):
    old = base.plan
    base.plan = plan
    try:
        return base.prepare(rt)
    finally:
        base.plan = old


def check_coverage(state):
    from .full_matrix import EmptyCertifiedBatch
    from .pair_runner import check_single_rx
    details = check_single_rx(state)
    data = state['U_coverage']
    if data['counts'].get('strong', 0) + data['counts'].get('weak', 0) == 0:
        raise EmptyCertifiedBatch('No source-U packet passed either tier in this batch')
    return dict(details, U_coverage=data)


def smoke(rt):
    from .full_matrix import EmptyCertifiedBatch
    from .runner import smoke as smoke_rows
    checks = []
    for row in (r for r in plan.candidate_rows() if r['model_seed'] == plan.SEEDS[0]):
        for offset in range(10):
            try:
                item = smoke_rows(rt, rows=lambda: [row], configure=plan.configuration, check_parent=plan.parent,
                                  scope=installed, check_u=check_coverage, graph_check=plan.check_graph,
                                  segment_output=True, labeled_batch_offset=offset, write_result=False)
            except EmptyCertifiedBatch:
                continue
            item[0]['source_labeled_batch_offset'] = offset
            checks.extend(item)
            break
        else:
            raise ValueError(f"No certified source-U packet in ten batches: {row['row_id']}")
    write(Path(rt['run_root']) / 'smoke.json', dict(status='PASS', checks=checks, real_checkpoint=True,
          optimizer_updates=0, query_reads=0, U_truth_reads=0))


def analyze(rt):
    from . import target_tuning_report as report
    old = report.plan
    report.plan = plan
    try:
        report.analyze(rt)
    finally:
        report.plan = old
    refresh_gate3(rt)


def refresh_gate3(rt):
    from . import target_tuning_report as report
    # The separately running full-weight 3/5 rows are never a dependency for
    # training or scoring this run. Record any already completed paired scores.
    root = Path(rt['run_root'])
    prior_root = Path(rt['project']) / 'runs' / '20261112-phase1-pamf-prototype-fullsource-tuning-m33-r01'
    prior_sources = read(prior_root / 'sources.json')
    current = read(root / 'analysis' / 'results.json')['rows']
    pairs, pending = [], []
    for row in current:
        seed = row['model_seed']
        key = f'ten_class-prototype-gate3-s{seed}'
        folder = Path(prior_sources[key]['scores'])
        if not (folder / 'complete.json').is_file():
            pending.append(key)
            continue
        gate3 = report.load_score(folder)
        if gate3['counts'] != row['current']['counts']:
            raise ValueError('Full-weight 3/5 and coverage candidates must have matched test populations')
        pairs.append(dict(row_id=row['row_id'], model_seed=seed, gate3=gate3,
                          clean_delta=row['current']['clean'] - gate3['clean'],
                          practical_delta=row['current']['practical_mean'] - gate3['practical_mean']))
    write(root / 'analysis' / 'gate3_comparison.json', dict(pairs=pairs, pending=sorted(set(pending)),
          interpretation='matched full-weight 3/5; previously exposed test set; exploratory'), exclusive=False)


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


def dispatch(rt):
    from experiments.pamf_branch_training import runner as queue, prepare as preparation
    from experiments.pamf_physics_branch.runner import admits as physical_admits
    from experiments.pamf_physics_route.runner import launch

    def admits(gpu, active, required_mib):
        if any(item['gpu'] == gpu for _, item in active.values()):
            return False
        return physical_admits(gpu, active, required_mib, max_total=2)

    old = preparation.prepare, queue.admits, queue.launch, queue.analyze
    preparation.prepare, queue.admits = prepare, admits
    queue.launch = partial(launch, module='experiments.pamf_physics_tri.coverage_runner')
    queue.analyze = analyze
    try:
        return queue.dispatch(dict(rt, prioritize_prediction=True))
    finally:
        preparation.prepare, queue.admits, queue.launch, queue.analyze = old


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['dispatch', 'worker', 'refresh_gate3'])
    parser.add_argument('--kind')
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = read(args.config)
    if args.action == 'dispatch':
        dispatch(config)
    elif args.action == 'refresh_gate3':
        refresh_gate3(config)
    else:
        worker(config, args.kind)
