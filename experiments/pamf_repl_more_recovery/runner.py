"""Recover only interrupted rows, preserving all completed source artifacts."""
import argparse
from functools import partial
from pathlib import Path
from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_repl_weight import runner as prior
from . import plan
from .objective import installed


def prepare(rt):
    from experiments.pamf_physics_tri import target_tuning_runner as common
    root, project, ctx = Path(rt['run_root']), Path(rt['project']), read(rt['context'])
    old_root = project/'runs'/plan.PREVIOUS_RUN
    old_state = read(old_root/'pipeline_state.json')
    if old_state['status'] != 'PARTIAL_TECHNICAL_FAILURE':
        raise ValueError('Prior owner must have finished preserving healthy workers')
    expected_failed = ['train__'+plan.RECOVER_E30]
    if old_state['failed_jobs'] != expected_failed:
        raise ValueError('Prior failure set differs from recovery matrix')
    for row in plan.candidate_rows():
        status = old_state['jobs']['train__'+row['row_id']]['status']
        expected = ('COMPLETE' if row['row_id'] in plan.REUSE_E50 else
                    'FAILED' if row['row_id'] == plan.RECOVER_E30 else 'QUEUED')
        if status != expected:
            raise ValueError('Prior row state differs: '+row['row_id'])
    old = common.plan; common.plan = plan
    try:
        jobs, sources, configs = common.make_jobs(root, project, ctx, rt['code_commit'], rt['frozen_sources'])
    finally:
        common.plan = old
    for rid in plan.REUSE_E50:
        train = configs['train__'+rid]
        source = plan.reused_e50(train)
        configs['predict__'+rid]['source_output'] = str(source)
        sources[rid]['source_output'] = str(source)
        for job in jobs:
            if job['id'] == 'predict__'+rid:
                job['dependencies'] = [str(source/'complete.json')]
    jobs = [j for j in jobs if not (j['kind'] == 'train' and j['id'].split('__',1)[1] in plan.REUSE_E50)]
    for job in jobs:
        write(job['config'], configs[job['id']])
    old_sources = read(project/'runs'/'20261027-phase1-pamf-paper-completion-m119-r03'/'sources.json')
    controls = {}
    for setting, pid in plan.SETTINGS.items():
        original = f'{setting}-factorial-ssl-full-permuted-s{plan.SEED}'
        controls[setting] = dict(tri=old_sources[original], pl=ctx['existing'][f'{pid}-ordinary_PL-s{plan.SEED}'])
    write(root/'jobs.json', jobs); write(root/'sources.json', sources); write(root/'controls.json', controls)
    return jobs


def smoke(rt):
    from experiments.pamf_physics_tri.runner import smoke as check_rows
    from experiments.pamf_repl_weight.recovery import recover_e30_telemetry
    def check_u(state):
        if not state['accepted'] or state['U_truth_read'] or state['U_gradient_guard']['calls'] != 1:
            raise ValueError('Original accepted-U and guard must run')
        return dict(accepted=state['accepted'])
    checks = check_rows(rt, rows=plan.candidate_rows, configure=plan.configuration, check_parent=plan.parent,
                        scope=installed, check_u=check_u, graph_check=plan.check_graph,
                        segment_output=True, write_result=False)
    ctx = read(rt['context'])
    failed = next(row for row in plan.candidate_rows() if row['row_id'] == plan.RECOVER_E30)
    c = plan.configuration(ctx, failed, rt['project'], rt['code_commit'])
    previous, payload = plan.recovered_e30(c)
    state = recover_e30_telemetry(previous['folder'], c['num_classes'])
    del payload
    write(Path(rt['run_root'])/'smoke.json', dict(status='PASS', checks=checks, real_checkpoint=True,
          recovered_e30_calls=state['calls'], optimizer_updates=0, query_reads=0, U_truth_reads=0))


def worker(c, kind):
    if kind == 'score':
        from experiments.pamf_paper_completion.evaluation import score
        return score(c)
    from experiments.pamf_repl_completion.runner import activate
    activate(c)
    if kind == 'smoke': return smoke(c)
    if kind == 'predict':
        from experiments.pamf_physics_tri.development_evaluation import predict
        return predict(c)
    if kind == 'train':
        if c['row_id'] == plan.RECOVER_E30:
            from experiments.pamf_repl_weight import recovery
            old = recovery.plan, recovery.installed
            recovery.plan, recovery.installed = plan, installed
            try: return recovery.train(c)
            finally: recovery.plan, recovery.installed = old
        from experiments.pamf_physics_tri.source import train_trajectory
        return train_trajectory(c, plan.parent(c), installed, ssl=True, physical_L=False)
    raise ValueError('Unknown recovery stage')


def analyze(rt):
    old = prior.plan; prior.plan = plan
    try: return prior.analyze(rt)
    finally: prior.plan = old


def dispatch(rt):
    from experiments.pamf_branch_training import runner as queue, prepare as preparation
    from experiments.pamf_physics_branch.runner import admits as physical_admits
    from experiments.pamf_physics_route.runner import launch
    def admits(gpu, active, mib):
        if sum(item['gpu'] == gpu for _, item in active.values()) >= 2:
            return False
        return physical_admits(gpu, active, max(mib, 10000), max_total=5)
    old = preparation.prepare, queue.admits, queue.launch, queue.analyze
    preparation.prepare, queue.admits = prepare, admits
    queue.launch = partial(launch, module='experiments.pamf_repl_more_recovery.runner')
    queue.analyze = analyze
    try: return queue.dispatch(dict(rt, prioritize_prediction=True))
    finally: preparation.prepare, queue.admits, queue.launch, queue.analyze = old


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['dispatch','worker']); p.add_argument('--kind'); p.add_argument('--config',required=True)
    a = p.parse_args(); c = read(a.config); dispatch(c) if a.action == 'dispatch' else worker(c, a.kind)
