"""Single owner: source continuation, clean prediction, independent scoring."""
import argparse
from functools import partial
from pathlib import Path

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_physics_tri import target_tuning_runner as common
from . import plan
from .objective import installed


def prepare(rt):
    root, ctx = Path(rt['run_root']), read(rt['context'])
    old = common.plan
    common.plan = plan
    try:
        jobs, sources, configs = common.make_jobs(root, rt['project'], ctx, rt['code_commit'], rt['frozen_sources'])
    finally:
        common.plan = old
    for job in jobs:
        write(job['config'], configs[job['id']])
    old_sources = read(Path(rt['project']) / 'runs' / '20261027-phase1-pamf-paper-completion-m119-r03' / 'sources.json')
    controls = {}
    for setting, pid in plan.SETTINGS.items():
        original = f'{setting}-factorial-ssl-full-permuted-s{plan.SEED}'
        controls[setting] = dict(tri=old_sources[original], pl=ctx['existing'][f'{pid}-ordinary_PL-s{plan.SEED}'])
    write(root / 'jobs.json', jobs)
    write(root / 'sources.json', sources)
    write(root / 'controls.json', controls)
    return jobs


def smoke(rt):
    from experiments.pamf_physics_tri.runner import smoke as check_rows
    from .recovery import proxy_checkpoint, recover_e30_telemetry
    def check_u(state):
        if not state['accepted'] or state['U_truth_read'] or state['U_gradient_guard']['calls'] != 1:
            raise ValueError('Expected original accepted-U and guard path')
        if sum(state['U_pseudo_observation']['accepted_by_pseudo_class'].values()) != state['accepted']:
            raise ValueError('Pseudo histogram and accepted counter differ')
        return dict(accepted=state['accepted'], guard=state['U_gradient_guard'])
    checks = check_rows(rt, rows=plan.candidate_rows, configure=plan.configuration, check_parent=plan.parent,
                        scope=installed, check_u=check_u, graph_check=plan.check_graph,
                        segment_output=True, write_result=False)
    ctx = read(rt['context'])
    recovered = []
    for row in plan.candidate_rows():
        c = plan.configuration(ctx, row, rt['project'], rt['code_commit'])
        previous, payload = plan.recovered_e30(c)
        stats = recover_e30_telemetry(previous['folder'], c['num_classes'])
        recovered.append(dict(row=row['row_id'], prior_calls=stats['calls'], seen=stats['seen']))
        if row['recipe'] == 'weight3':
            proxy_checkpoint(c, payload, Path(rt['run_root'])/'smoke_e30_v'/row['setting'])
    write(Path(rt['run_root']) / 'smoke.json', dict(status='PASS', checks=checks, real_checkpoint=True,
          recovered_e30=recovered, source_v_proxy_settings=2, optimizer_updates=0, query_reads=0, U_truth_reads=0))


def worker(c, kind):
    if kind == 'score':
        from experiments.pamf_paper_completion.evaluation import score
        return score(c)
    from experiments.pamf_repl_completion.runner import activate
    activate(c)
    if kind == 'smoke':
        return smoke(c)
    if kind == 'train':
        from .recovery import train
        return train(c)
    if kind == 'predict':
        from experiments.pamf_physics_tri.development_evaluation import predict
        return predict(c)
    raise ValueError('Unknown stage')


def pooled(folder):
    folder = Path(folder)
    if read(folder / 'complete.json')['status'] != 'SCORED_COMPLETE':
        raise ValueError('Scoring incomplete')
    data = read(folder / 'scores.json')
    if not data.get('truth_last'):
        raise ValueError('Independent truth-last scoring required')
    rows = [r for r in data['results'] if r['dimension'] == 'overall' and r['readout'] == 'raw'
            and r['view'] == 'clean' and r['test'] in ('cross_rx', 'cross_both')]
    if len(rows) != 2 or {r['test'] for r in rows} != {'cross_rx', 'cross_both'}:
        raise ValueError('Expected exactly both clean target populations')
    count = sum(r['count'] for r in rows)
    return dict(count=count, accuracy=sum(round(r['accuracy'] * r['count']) for r in rows) / count,
                per_test={r['test']: r['accuracy'] for r in rows})


def analyze(rt):
    root = Path(rt['run_root'])
    sources, controls = read(root/'sources.json'), read(root/'controls.json')
    rows = []
    for row in plan.candidate_rows():
        cand = pooled(sources[row['row_id']]['scores'])
        ctrl = controls[row['setting']]
        tri, pl = pooled(ctrl['tri']['scores']), pooled(ctrl['pl']['scores'])
        if not cand['count'] == tri['count'] == pl['count']:
            raise ValueError('Control physical population counts differ')
        rows.append(dict(**row, candidate=cand, original_tri=tri, ordinary_pl=pl,
                         delta_tri_pp=100*(cand['accuracy']-tri['accuracy']),
                         delta_pl_pp=100*(cand['accuracy']-pl['accuracy'])))
    out = root/'analysis'; out.mkdir(exist_ok=False)
    write(out/'results.json', dict(status='ANALYZED', rows=rows, seed_selected_using_prior_target=True,
          interpretation='previously exposed target and selected seed; exploratory, not independent confirmation',
          test_scope='raw clean only', target_feedback_forbidden=True))
    write(out/'complete.json', dict(status='ANALYZED', rows=len(rows), test_scope='raw clean only'))


def dispatch(rt):
    from experiments.pamf_branch_training import runner as queue, prepare as preparation
    from experiments.pamf_physics_branch.runner import admits as physical_admits
    from experiments.pamf_physics_route.runner import launch
    old = preparation.prepare, queue.admits, queue.launch, queue.analyze
    preparation.prepare = prepare
    queue.admits = lambda gpu, active, mib: physical_admits(gpu, active, mib, max_total=5)
    queue.launch = partial(launch, module='experiments.pamf_repl_weight.runner')
    queue.analyze = analyze
    try:
        return queue.dispatch(dict(rt, prioritize_prediction=True))
    finally:
        preparation.prepare, queue.admits, queue.launch, queue.analyze = old


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['dispatch', 'worker'])
    parser.add_argument('--kind')
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = read(args.config)
    dispatch(config) if args.action == 'dispatch' else worker(config, args.kind)
