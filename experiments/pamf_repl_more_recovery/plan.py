"""Additional single-factor candidates from original same-setting E20."""
from copy import deepcopy
from pathlib import Path
from experiments.cric_dura_comparison.io import read
from experiments.pamf_paper_completion import plan as paper

RUN = '20261115-phase1-pamf-repl-more-recovery-m10-r02'
PREVIOUS_RUN = '20261115-phase1-pamf-repl-more-clean-m16-r01'
SEED = 2026092701
REUSE_E50 = {'ten_class-repl-style0-s2026092701', 'ten_class-repl-kl05-s2026092701', 'source_rx_6-repl-weight075-s2026092701', 'source_rx_6-repl-shallow0-s2026092701', 'source_rx_6-repl-shallow025-s2026092701'}
RECOVER_E30 = 'source_rx_6-repl-weight45-s2026092701'
SETTINGS = {'ten_class': 'ManyRx', 'source_rx_6': 'm6-natural'}
RECIPES = {'weight075': {'weight': .75}, 'weight45': {'weight': 4.5},
    'shallow0': {'shallow_factor': 0.}, 'shallow025': {'shallow_factor': .25},
    'joint0': {'joint_weight': 0.}, 'extrap0': {'extrap_coefficient': 0.},
    'style0': {'style_probability': 0.}, 'kl05': {'kl_factor': .5}}


def candidate_rows():
    return [dict(row_id=f'{setting}-repl-{name}-s{SEED}', setting=setting, protocol_id=pid,
                 model_seed=SEED, recipe=name, views=['clean'])
            for setting, pid in SETTINGS.items() for name in RECIPES
            if setting == 'source_rx_6' or name in ('style0', 'kl05')]


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
    c.update(run_id=RUN,row_id=row['row_id'],code_commit=commit,algorithm_id='pamf_repl_more_recovery_clean_v1',
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


def reused_e50(c):
    """Check the same row's frozen source, never a target-selected model."""
    row = validate(c)
    if row['row_id'] not in REUSE_E50:
        raise ValueError('Row is not registered for E50 reuse')
    root = Path(c['output_root'])
    if root.name in ('segment30', 'segment50'):
        root = root.parent
    source = root.parents[3] / PREVIOUS_RUN / 'F' / row['row_id'] / 'source'
    done = read(source / 'complete.json')
    prior = done['config']
    if (done['status'] != 'SOURCE_FROZEN' or done['epoch'] != 50 or done['updates'] != 11100
            or done['target_access'] or done['U_truth_read'] or done['head_fit_updates']
            or prior['run_id'] != PREVIOUS_RUN or prior['row_id'] != row['row_id']
            or read(prior['source_contract']) != read(c['source_contract'])):
        raise ValueError('Frozen source E50 identity or permissions mismatch')
    for key in ('tuning_recipe', 'hard_pl', 'hard_pl_mixstyle', 'hard_pl_gradient_guard',
                'hard_pl_joint_contrast', 'hard_pl_receiver_extrapolation', 'source_cache',
                'source_rxs', 'source_days', 'paper_row', 'shared_prefix'):
        if prior[key] != c[key]:
            raise ValueError('E50 candidate differs: '+key)
    if not (source/'final_ssdg.pth').is_file():
        raise FileNotFoundError('Frozen E50 checkpoint missing')
    return source


def recovered_e30(c):
    """Reuse only this candidate's already-complete E30 after the source-validation OOM."""
    import torch
    row = validate(c)
    prefix = parent(c)
    root = Path(c['output_root'])
    if root.name in ('segment30', 'segment50'):
        root = root.parent
    previous = root.parents[3] / PREVIOUS_RUN / 'F' / row['row_id'] / 'source' / 'segment30'
    done = read(previous / 'complete.json')
    prior = done['config']
    if (done['status'] != 'SOURCE_FROZEN' or done['epoch'] != 10 or done['updates'] != 2220
            or done['target_access'] or done['U_truth_read'] or done['head_fit_updates']
            or prior['run_id'] != PREVIOUS_RUN or prior['row_id'] != row['row_id']
            or prior['parent_source'] != prefix['folder'] or prior['step_offset'] != 4440
            or read(prior['source_contract']) != read(c['source_contract'])):
        raise ValueError('Completed E30 source ancestry or permissions mismatch')
    for key in ('tuning_recipe', 'hard_pl', 'hard_pl_mixstyle', 'hard_pl_gradient_guard',
                'hard_pl_joint_contrast', 'hard_pl_receiver_extrapolation', 'source_cache', 'paper_row'):
        if prior[key] != c[key]:
            raise ValueError('E30 candidate differs: ' + key)
    cp = torch.load(previous/'final_ssdg.pth', map_location='cpu', weights_only=False)
    if (cp['run_id'] != PREVIOUS_RUN or cp['candidate_id'] != row['row_id'] or cp['epoch'] != 10
            or cp['a1_ema_successful_updates'] != 2220 or cp['args']['teacher_ckpt']
            or cp['args']['baseline_ckpt'] != str(Path(prefix['folder'])/'final_ssdg.pth')):
        raise ValueError('E30 checkpoint does not match its source trajectory')
    return dict(folder=str(previous), **done), cp

