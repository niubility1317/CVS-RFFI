"""Full-source continuations, each followed by seven-view truth-last scoring."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
from functools import partial
from pathlib import Path

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_single_view_paper.plan import TASKS
from . import target_tuning_plan as plan
from .practical_gate import installed as practical_scope
from .source_prototype import SourcePrototype


@contextmanager
def installed(c, holder):
    from experiments.pamf_paper_completion.model import installed as paper_scope

    @contextmanager
    def adapted(cfg, state, enabled, objective_factory):
        with paper_scope(cfg, state, ssl_enabled=enabled, objective_factory=objective_factory,
                         validate=plan.validate):
            yield

    with practical_scope(c, holder, base_scope=adapted, objective_factory=SourcePrototype):
        yield


def make_jobs(run_root, project, ctx, commit, frozen_sources):
    root = Path(run_root)
    jobs, sources, configs = [], {}, {}
    for row in plan.candidate_rows():
        rid, pid = row['row_id'], row['protocol_id']
        h = ctx['handles'][pid]
        train = plan.configuration(ctx, row, project, commit)
        train['frozen_sources'] = frozen_sources
        source = Path(train['output_root'])
        out = root / 'readouts' / rid
        pred = dict(run_id=plan.RUN, row_id=rid, paper_row=train['paper_row'], protocol_id=pid,
                    model_seed=row['model_seed'], source_output=str(source), source_contract=h['source_contract'],
                    classes=h['classes'], source_days=[1, 2, 3],
                    tests={task: dict(views_root=h['tests'][task]['views_root']) for task in TASKS},
                    restored_caches={}, views=row['views'], readouts=['raw'], reuse_predictions=None,
                    output_root=str(out / 'predictions'), frozen_sources=frozen_sources)
        score = deepcopy(pred)
        score.update(prediction_output=pred['output_root'], output_root=str(out / 'scores'),
                     tests={task: h['tests'][task] for task in TASKS})
        for key in ('source_output', 'restored_caches', 'frozen_sources', 'reuse_predictions'):
            score.pop(key, None)
        for kind, cfg, completion, status, dependencies in (
            ('train', train, source / 'complete.json', 'SOURCE_FROZEN', []),
            ('predict', pred, out / 'predictions' / 'complete.json', 'ALL_PREDICTIONS_COMPLETE', [source / 'complete.json']),
            ('score', score, out / 'scores' / 'complete.json', 'SCORED_COMPLETE', [out / 'predictions' / 'complete.json'])):
            jid = kind + '__' + rid
            configs[jid] = cfg
            jobs.append(dict(id=jid, kind=kind, config=str(root / 'configs' / (jid + '.json')),
                             completion=str(completion), successful_statuses=[status],
                             dependencies=[str(item) for item in dependencies]))
        sources[rid] = dict(source_output=str(source), predictions=pred['output_root'], scores=score['output_root'])
    return jobs, sources, configs


def prepare(rt):
    root, ctx = Path(rt['run_root']), read(rt['context'])
    jobs, sources, configs = make_jobs(root, rt['project'], ctx, rt['code_commit'], rt['frozen_sources'])
    for job in jobs:
        write(job['config'], configs[job['id']])
    prior = read(Path(rt['project']) / 'runs' / '20261112-phase1-pamf-tri-two-methods-full-m60-r02' / 'sources.json')
    controls = {}
    for row in plan.candidate_rows():
        key = f"{row['protocol_id']}-ordinary_PL-s{row['model_seed']}"
        controls[key] = ctx['existing'][key]
        old = plan.baseline_row(row)['row_id']
        controls[old] = prior[old]
    write(root / 'jobs.json', jobs)
    write(root / 'sources.json', sources)
    write(root / 'controls.json', controls)
    return jobs


def smoke(rt):
    from .full_matrix import EmptyCertifiedBatch
    from .pair_runner import check_full_matrix
    from .runner import smoke as smoke_rows

    checks = []
    for row in (r for r in plan.candidate_rows() if r['model_seed'] == plan.SEEDS[0]):
        for offset in range(10):
            try:
                item = smoke_rows(rt, rows=lambda: [row], configure=plan.configuration, check_parent=plan.parent,
                                  scope=installed, check_u=check_full_matrix, graph_check=plan.check_graph,
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
    from experiments.pamf_branch_training import runner as base, prepare as preparation
    from experiments.pamf_physics_branch.runner import admits as physical_admits
    from experiments.pamf_physics_route.runner import launch
    from .target_tuning_report import analyze

    def admits(gpu, active, required_mib):
        if any(item['gpu'] == gpu for _, item in active.values()):
            return False
        return physical_admits(gpu, active, required_mib, max_total=2)

    old = preparation.prepare, base.admits, base.launch, base.analyze
    preparation.prepare, base.admits = prepare, admits
    base.launch = partial(launch, module='experiments.pamf_physics_tri.target_tuning_runner')
    base.analyze = analyze
    try:
        return base.dispatch(dict(rt, prioritize_prediction=True))
    finally:
        preparation.prepare, base.admits, base.launch, base.analyze = old


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['dispatch', 'worker'])
    parser.add_argument('--kind')
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    c = read(args.config)
    dispatch(c) if args.action == 'dispatch' else worker(c, args.kind)
