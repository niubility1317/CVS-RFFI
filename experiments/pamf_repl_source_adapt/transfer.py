"""Validate a frozen source decision at E50 on matched full protocols and seeds."""
import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from functools import partial
from pathlib import Path
from types import SimpleNamespace

from experiments.cric_dura_comparison.io import read, write
from .controller import choose_weight
from .pilot import RUN as DEVELOPMENT_RUN
from .pilot import SETTINGS, SEED, WEIGHTS, collect_evidence, paper


RUN = '20261119-phase1-pamf-repl-sourceadapt-calibration-clean-m12-r01'


def calibration_decisions(root):
    root = Path(root)
    state = read(root/'pipeline_state.json')
    old = read(root/'analysis'/'source_decisions.json')
    if state['status'] != 'ANALYZED' or old['target_access'] or old['U_truth_read'] or old['selected_seed'] != SEED:
        raise ValueError('Completed original source development matrix required')
    folds, evidence = collect_evidence(root, SETTINGS)
    decisions = {}
    for name in SETTINGS:
        choice = choose_weight(folds[name], evidence[name], noise_aware=True)
        choice['candidates'] = {str(weight): metrics for weight, metrics in choice['candidates'].items()}
        decisions[name] = choice
    return dict(status='SOURCE_ADAPTATION_DECISIONS_FROZEN', selected_seed=SEED, target_access=False,
                U_truth_read=False, policy='development_calibration',
                decisions=decisions)


def selected_rows(evidence):
    if (evidence['status'] != 'SOURCE_ADAPTATION_DECISIONS_FROZEN' or evidence['selected_seed'] != SEED
            or evidence['target_access'] or evidence['U_truth_read'] or evidence['policy'] != 'development_calibration'
            or set(evidence['decisions']) != set(SETTINGS)):
        raise ValueError('Registered frozen source decisions required for development calibration')
    result = []
    for setting in SETTINGS:
        decision = evidence['decisions'][setting]
        weight = decision['weight']
        if (weight not in WEIGHTS or (weight == 6. and decision['decision'] != 'retain_native')
                or (weight != 6. and (decision['decision'] != 'source_improvement'
                                     or not decision['candidates'][str(weight)]['eligible']))):
            raise ValueError('Dose must match the source-only decision')
        for seed in paper.SEEDS:
            result.append(dict(row_id=f'{setting}-tri-source-adapt-s{seed}', setting=setting,
                               protocol_id=paper.PROTOCOLS[setting][0], model_seed=seed, weight=weight,
                               views=['clean'], reuses_original=weight == 6.))
    return result


def original_row(row):
    return f"{row['setting']}-factorial-ssl-full-permuted-s{row['model_seed']}"


def source_configuration(ctx, row, project, commit, run_id):
    from experiments.pamf_physics_tri.plan import continuation

    if row['setting'] == 'main_split':
        cfg = continuation(ctx, row, str(project), commit, run_id)
    else:
        previous = Path(project)/'runs'/paper.RUN/'F'/original_row(row)/'source'
        done = read(previous/'complete.json')
        cfg = deepcopy(done['config'])
        paper.validate_config(cfg)
        if (done['status'] != 'SOURCE_FROZEN' or done['epoch'] != 50 or done['updates'] != 11100
                or done['target_access'] or done['U_truth_read']
                or read(cfg['source_contract']) != read(ctx['handles'][row['protocol_id']]['source_contract'])):
            raise ValueError('Exactly matched historical source model required')
        cfg['shared_prefix'] = str(previous/'segment20')
    cfg.update(run_id=run_id, row_id=row['row_id'], code_commit=commit, source_adaptation=row,
               output_root=str(Path(project)/'runs'/run_id/'F'/row['row_id']/'source'))
    cfg['hard_pl']['weight'] = row['weight']
    cfg['hard_pl_gradient_guard']['weight'] = row['weight']
    return cfg


def configure(ctx, row, project, commit):
    cfg = source_configuration(ctx, row, project, commit, RUN)
    validate(cfg)
    return cfg


def validate(c):
    row = c['source_adaptation']
    if (c['run_id'] != RUN or c['row_id'] != row['row_id'] or c['model_seed'] != row['model_seed']
            or c['paper_row']['row_id'] != original_row(row) or c['protocol_id'] != row['protocol_id']
            or c['optimizer_updates'] != 11100 or c['epochs'] != 50
            or row['weight'] not in WEIGHTS[1:] or c['hard_pl']['weight'] != row['weight']
            or c['hard_pl_gradient_guard']['weight'] != row['weight'] or c['hard_pl']['threshold'] != .95
            or any(c.get(key) for key in ('tests', 'target_truth', 'target_inputs', 'views_root', 'checkpoint', 'source_fold'))):
        raise ValueError('Full source continuation must use the preregistered dose alone')


def parent(c, *, validate_config=validate):
    from experiments.pamf_branch_training.source import checked_e20_parent
    from experiments.pamf_repl_view_shuffle import plan as shuffled

    validate_config(c)
    evidence = checked_e20_parent(c)
    prior = evidence['config']
    row = c['source_adaptation']
    if row['setting'] == 'main_split':
        shuffled.validate_config(prior)
        matches = prior['row_id'] == f"main_split-repl-shuffled_time-3view-s{row['model_seed']}"
    else:
        matches = prior['run_id'] == paper.RUN and prior['row_id'] == original_row(row) and prior['paper_row'] == c['paper_row']
    if not matches or prior.get('source_fold'):
        raise ValueError('Only full original same-setting same-seed E20 can enter target verification')
    return evidence


@contextmanager
def installed(c, holder):
    from experiments.pamf_paper_completion.model import installed as paper_scope
    from experiments.pamf_repl_view_shuffle.source import installed as shuffled_scope

    validate(c)
    scope = shuffled_scope(c, holder, True) if c['source_adaptation']['setting'] == 'main_split' else paper_scope(c, holder, ssl_enabled=True, validate=validate)
    with scope:
        yield


def prepare_jobs(rt, rows, configurator):
    from experiments.pamf_physics_tri import target_tuning_runner as standard

    root = Path(rt['run_root'])
    project = Path(rt['project'])
    ctx = read(rt['context'])
    changed = [row for row in rows if not row['reuses_original']]
    previous = standard.plan
    standard.plan = SimpleNamespace(RUN=rt['run_id'], candidate_rows=lambda: changed, configuration=configurator)
    try:
        jobs, sources, configs = standard.make_jobs(root, project, ctx, rt['code_commit'], rt['frozen_sources'])
    finally:
        standard.plan = previous
    old = read(project/'runs'/paper.RUN/'sources.json')
    controls = {}
    for row in rows:
        rid, pid, seed = row['row_id'], row['protocol_id'], row['model_seed']
        controls[rid] = dict(tri=old[original_row(row)], pl=ctx['existing'][f'{pid}-ordinary_PL-s{seed}'])
        if row['reuses_original']:
            sources[rid] = controls[rid]['tri']
    for job in jobs:
        write(job['config'], configs[job['id']])
    write(root/'jobs.json', jobs)
    write(root/'rows.json', rows)
    write(root/'sources.json', sources)
    write(root/'controls.json', controls)
    return jobs


def prepare(rt):
    decision = calibration_decisions(Path(rt['project'])/'runs'/DEVELOPMENT_RUN)
    write(Path(rt['run_root'])/'calibration_source_decisions.json', decision)
    return prepare_jobs(rt, selected_rows(decision), configure)


def smoke(rt):
    from experiments.pamf_physics_tri.runner import smoke as standard
    from experiments.pamf_physics_tri.full_matrix_plan import check_graph

    rows = [row for row in read(Path(rt['run_root'])/'rows.json') if not row['reuses_original']]
    if not rows:
        raise ValueError('No source decision activated; exact original results should be reused directly')
    standard(rt, rows=lambda: rows, configure=configure, check_parent=parent, scope=installed,
             check_u=lambda state: dict(accepted=state['accepted']), graph_check=check_graph, segment_output=True)


def analyze(rt, *, interpretation='retrospective source-rule calibration on design cohort', source_decision_run=DEVELOPMENT_RUN):
    import numpy as np
    from experiments.pamf_repl_weight.analyze_paired import compare, prediction
    from experiments.pamf_repl_weight.runner import pooled

    root = Path(rt['run_root'])
    rows, sources, controls = [read(root/f'{name}.json') for name in ('rows', 'sources', 'controls')]
    ctx = read(rt['context'])
    output = []
    for row in rows:
        rid = row['row_id']
        items = dict(adaptive=sources[rid], **controls[rid])
        scores = {name: pooled(item['scores']) for name, item in items.items()}
        counts, splits = {}, {}
        for key in controls[rid]:
            c = Counter()
            splits[key] = {}
            for task in ('cross_rx', 'cross_both'):
                ids, current = prediction(items['adaptive']['predictions'], task)
                baseline_ids, baseline = prediction(items[key]['predictions'], task)
                if not np.array_equal(ids, baseline_ids):
                    raise ValueError('All clean evaluations must use the same physical records')
                truth = read(ctx['handles'][row['protocol_id']]['tests'][task]['truth'])
                per_test = compare(ids, current, baseline, truth)['overall']
                splits[key][task] = per_test
                c.update(per_test)
            if (c['count'] != scores['adaptive']['count'] or c['count'] != scores[key]['count']
                    or not np.isclose(c['candidate_correct']/c['count'], scores['adaptive']['accuracy'])
                    or not np.isclose(c['baseline_correct']/c['count'], scores[key]['accuracy'])):
                raise ValueError('Physical pair analysis disagrees with independent clean scores')
            counts[key] = dict(c)
        output.append(dict(**row, scores=scores, paired=counts, by_test=splits))
    write(root/'analysis'/'results.json', dict(status='TRANSFER_ANALYZED', rows=output,
          source_decision_run=source_decision_run, interpretation=interpretation,
          target_feedback=False))


def dispatch(rt, *, prepare_fn=prepare, analyze_fn=analyze, module='experiments.pamf_repl_source_adapt.transfer'):
    from experiments.pamf_branch_training import prepare as prep, runner
    from experiments.pamf_physics_branch.runner import admits
    from experiments.pamf_physics_route.runner import launch

    original = prep.prepare, runner.admits, runner.launch, runner.analyze
    prep.prepare, runner.analyze = prepare_fn, analyze_fn
    runner.admits = lambda gpu, jobs, mib: admits(gpu, jobs, max(12288, mib), max_total=2)
    runner.launch = partial(launch, module=module)
    try:
        runner.dispatch(dict(rt, prioritize_prediction=True))
    finally:
        prep.prepare, runner.admits, runner.launch, runner.analyze = original


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('dispatch', 'worker'))
    parser.add_argument('--kind')
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    c = read(args.config)
    if args.action == 'dispatch':
        dispatch(c)
    elif args.kind == 'score':
        from experiments.pamf_paper_completion.evaluation import score
        score(c)
    else:
        from experiments.pamf_repl_completion.runner import activate
        activate(c)
        if args.kind == 'smoke':
            smoke(c)
        elif args.kind == 'train':
            from experiments.pamf_physics_tri.source import train_trajectory
            train_trajectory(c, parent(c), installed, ssl=True, physical_L=False)
        elif args.kind == 'predict':
            from experiments.pamf_physics_tri.development_evaluation import predict
            predict(c)
        else:
            parser.error('Unknown registered stage')
