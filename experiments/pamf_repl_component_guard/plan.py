"""Fixed source-anchored component projection candidates from original E20."""
from copy import deepcopy
from pathlib import Path
from experiments.cric_dura_comparison.io import read
from experiments.pamf_paper_completion import plan as paper

RUN = '20261115-phase1-pamf-component-guard-clean-m6-r01'
SEED = 2026092701
SETTINGS = {'ten_class': 'ManyRx', 'source_rx_6': 'm6-natural'}
RECIPES = {'shallow_project': {'project_components': ['shallow']},
    'extrap_project': {'project_components': ['extrap_delta']},
    'both_project': {'project_components': ['shallow', 'extrap_delta']}}


def candidate_rows():
    return [dict(row_id=f'{setting}-repl-{name}-s{SEED}', setting=setting, protocol_id=pid,
                 model_seed=SEED, recipe=name, views=['clean'])
            for setting, pid in SETTINGS.items() for name in RECIPES]


def apply_recipe(c, name):
    c=deepcopy(c); r=RECIPES[name]
    if 'weight' in r:
        c['hard_pl']['weight']=r['weight'];c['hard_pl_gradient_guard']['weight']=r['weight']
    if 'joint_weight' in r:c['hard_pl_joint_contrast']['weight']=r['joint_weight']
    if 'extrap_coefficient' in r:
        c['hard_pl_receiver_extrapolation']['extrapolated_CE_coefficient']=r['extrap_coefficient']
        c['hard_pl_receiver_extrapolation']['ordinary_CE_coefficient']=.5-r['extrap_coefficient']
    if 'style_probability' in r:c['hard_pl_mixstyle']['probability_per_level']=r['style_probability']
    return c


def configuration(ctx, row, project, commit, *, run_id=RUN):
    if run_id != RUN or row not in candidate_rows():raise ValueError('Unknown additional candidate')
    original=f"{row['setting']}-factorial-ssl-full-permuted-s{SEED}"
    source=Path(project)/'runs'/paper.RUN/'F'/original/'source'
    done=read(source/'complete.json')
    if done['status']!='SOURCE_FROZEN' or done['target_access'] or done['U_truth_read'] or done['epoch']!=50:
        raise ValueError('Original source not frozen or has target ancestry')
    old=deepcopy(done['config']);paper.validate_config(old)
    if read(old['source_contract'])!=read(ctx['handles'][row['protocol_id']]['source_contract']):
        raise ValueError('Source data roles mismatch')
    c=apply_recipe(old,row['recipe'])
    for k in ('parent_source','segment_epochs','epoch_offset','step_offset'):c.pop(k,None)
    c.update(run_id=RUN,row_id=row['row_id'],code_commit=commit,algorithm_id='pamf_repl_component_guard_clean_v1',
        tuning_row=deepcopy(row),tuning_recipe=deepcopy(RECIPES[row['recipe']]),
        shared_prefix=str(source/'segment20'),output_root=str(Path(project)/'runs'/RUN/'F'/row['row_id']/'source'))
    validate(c);return c


def validate(c):
    row=next((r for r in candidate_rows() if r['row_id']==c.get('row_id')),None)
    if row is None or c.get('tuning_row')!=row or c.get('tuning_recipe')!=RECIPES[row['recipe']]:
        raise ValueError('Additional candidate not registered')
    root=Path(c['output_root']);root=root.parent if root.name in ('segment30','segment50') else root
    if root.parts[-5:]!=('runs',RUN,'F',row['row_id'],'source') or c['run_id']!=RUN:
        raise ValueError('Independent candidate output required')
    if c['model_seed']!=SEED or c['protocol_id']!=row['protocol_id']:
        raise ValueError('Seed/protocol mismatch')
    if any(c.get(k) for k in ('target_inputs','target_truth','truth','tests','views_root','checkpoint')):
        raise ValueError('Target or truth in training configuration')
    return row


def parent(c):
    from experiments.pamf_branch_training.source import checked_e20_parent
    row=validate(c);root=Path(c['output_root']);root=root.parent if root.name in ('segment30','segment50') else root
    original=f"{row['setting']}-factorial-ssl-full-permuted-s{SEED}"
    expected=root.parents[3]/paper.RUN/'F'/original/'source'/'segment20'
    if Path(c['shared_prefix'])!=expected:raise ValueError('Exact same-contract E20 required')
    evidence=checked_e20_parent(c);prior=evidence['config'];paper.validate_config(prior)
    if prior['run_id']!=paper.RUN or prior['row_id']!=original or prior['paper_row']!=c['paper_row']:
        raise ValueError('Source ancestry or architecture mismatch')
    wanted=apply_recipe(prior,row['recipe'])
    for key in ('hard_pl','hard_pl_mixstyle','hard_pl_gradient_guard','hard_pl_joint_contrast',
                'hard_pl_receiver_extrapolation','source_rxs','source_days','source_cache'):
        if c[key]!=wanted[key]:raise ValueError('Candidate or source mismatch: '+key)
    return evidence


def check_graph(model,c):
    from experiments.pamf_physics_tri.full_matrix_plan import check_graph as check
    return check(model,c)
