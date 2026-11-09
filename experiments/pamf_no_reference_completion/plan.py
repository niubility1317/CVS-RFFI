"""One fixed matrix, distinct source recipes and explicit historical reuse."""
from copy import deepcopy
from experiments.pamf_single_view_paper.plan import PROTOCOLS, VIEWS

RUN = '20261109-phase1-pamf-no-reference-completion-m190-r01'
CE_VARIANT = 'pamf_tri_sigtime_no_reference_ce'
SEEDS = (2026092701, 2026092702, 2026092703)
EXTENDED_SEEDS = tuple(range(2026092701, 2026092709))
TASKS = ('cross_rx', 'cross_both')
READOUTS = ('raw', 'DURA-L', 'DURA-LU')
STRUCTURES = ('single_view', 'no_time_coordinate', 'moment', 'shuffled_time')


def rows():
    out = []
    def add(setting, method, seed, family='factorial', structure='no_reference', reuse=None):
        pid, display, _ = PROTOCOLS[setting]
        ssl = method == 'ssl'
        sup = method.startswith('sup') or method == 'ce'
        aug = family!='architecture' and method not in ('sup_noaug', 'pl_noaug')
        permuted = ssl or structure == 'shuffled_time'
        row_id = f'{setting}-{family}-{method}-{structure}-s{seed}'
        out.append(dict(row_id=row_id, data_setting=setting, protocol_id=pid, split_name=display,
                        family=family, method=method, structure=structure, permuted=permuted,
                        signature_views=1 if structure=='single_view' else 3,
                        channel_augmentation=aug, learning_regime='supervised' if sup else 'ssl' if ssl else 'pl',
                        model_seed=seed, epochs=200 if family=='architecture' else 50,
                        updates=10000 if family=='architecture' else 11100,
                        views=['clean'] if family=='architecture' else list(VIEWS),
                        readouts=['raw'] if family in ('structure_ce','architecture') else list(READOUTS),
                        reuse_key=reuse, new_training=reuse is None))
    for arm in STRUCTURES:
        for seed in SEEDS:
            add('main_split','ce',seed,'structure_ce',arm,
                f'single/{seed}' if arm=='single_view' else None)
    for setting in PROTOCOLS:
        for method in ('sup','pl','sup_noaug','pl_noaug','ssl'):
            for seed in SEEDS:
                add(setting,method,seed,reuse=f'ssl/{seed}' if setting=='main_split' and method=='ssl' else None)
    for seed in EXTENDED_SEEDS[3:]:
        add('main_split','ssl',seed)
    for seed in EXTENDED_SEEDS:
        add('main_split','ce',seed,'architecture')
    assert len(out)==190 and len({r['row_id'] for r in out})==190
    assert sum(not r['new_training'] for r in out)==6
    return out


def configuration(context, row, project, commit):
    if row not in rows() or not row['new_training']:
        raise ValueError('Only the registered new source row is configurable')
    if row['family']=='structure_ce':
        c=deepcopy(context['baselines'][str(row['model_seed'])]['complete']['config'])
        c.update(run_id=RUN,row_id=row['row_id'],code_commit=commit,output_root=project+'/runs/'+RUN+'/F/'+row['row_id']+'/source',
                 algorithm_id='pamf_no_reference_structure_ce',paper_method='PAMF CE structural control',
                 paper_row=deepcopy(row),shared_prefix=None)
        c['repl_row'].update(row_id=row['row_id'],method='NR_'+row['structure'],structure='NR_'+row['structure'],
                             reuse=False,readouts=['raw'])
        return validate(c)
    from experiments.pamf_paper_completion import plan as old
    ctx=deepcopy(context)
    if str(row['model_seed']) not in ctx['templates']:
        ctx['templates'][str(row['model_seed'])]=deepcopy(ctx['templates'][str(SEEDS[0])])
    original=(old.RUN,old.make_plan,old.CE_VARIANT)
    old.RUN,old.make_plan,old.CE_VARIANT=RUN,lambda:dict(rows=rows()),CE_VARIANT
    try:
        c=old.configuration(ctx,row,project,commit)
    finally:
        old.RUN,old.make_plan,old.CE_VARIANT=original
    if row['family']!='architecture':
        c['channel_augmentation']=row['channel_augmentation']
    return validate(c)


def validate(c):
    row=c.get('paper_row'); expected=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
    if row!=expected or expected is None or not row['new_training'] or c.get('run_id')!=RUN or c.get('model_seed')!=row['model_seed']:
        raise ValueError('Registered model and seed required')
    if any(c.get(k) for k in ('truth','target_truth','views_root','tests','target_inputs','checkpoint','teacher_checkpoint','shared_prefix')):
        raise ValueError('Only same-contract scratch source inputs permitted')
    if row['family']=='architecture':
        if (c.get('variant')!=CE_VARIANT or c.get('epochs')!=200 or c.get('augmentation') or
                c.get('extra_losses') or c.get('selection')!='fixed_last_epoch'):
            raise ValueError('Fixed E200 pure CE required')
        return c
    if (c.get('source_days')!=[1,2,3] or c.get('num_classes')!=(10 if row['data_setting']=='ten_class' else 6) or
            c.get('own_EMA') or c.get('native_U_gate') or c.get('posthoc_head_fit') or
            c.get('epochs')!=50 or c.get('optimizer_updates')!=11100 or c.get('trajectory')!=[20,10,20]):
        raise ValueError('Fixed source split, model budget and permissions required')
    if row['family']=='structure_ce':
        if (c.get('ablation')!='F15' or c.get('U_use') or c.get('satellite_CE_weight')!=.68 or
                c['repl_row']['structure']!='NR_'+row['structure'] or row['permuted']!=(row['structure']=='shuffled_time')):
            raise ValueError('CE structural control differs')
    elif (c.get('U_use')!=(row['learning_regime']!='supervised') or
          c.get('channel_augmentation')!=row['channel_augmentation'] or
          row['structure']!='no_reference' or row['permuted']!=(row['learning_regime']=='ssl')):
        raise ValueError('Main method or graph differs')
    return c
