"""The user-approved, fixed 226-model logical matrix."""
from experiments.pamf_single_view_paper.plan import PROTOCOLS, VIEWS
from copy import deepcopy

RUN = '20261109-phase1-pamf-split-spectrum-paper-m226-r01'
SEEDS = (2026092701, 2026092702, 2026092703)
EXTENDED_SEEDS = tuple(range(2026092701, 2026092709))
TASKS = ('cross_rx', 'cross_both')
READOUTS = ('raw', 'DURA-L', 'DURA-LU')
CE_VARIANT = 'pamf_split_spectrum_ce'

# time, path, PA IQ, original PA spectral statistics, exact historical arm
BRANCHES = {
    'full': (1, 1, 1, 1, 'strict_no_frequency'),
    'no_spectrum': (1, 1, 1, 0, None),
    'no_pa_iq': (1, 1, 0, 1, None),
    'no_time': (0, 0, 1, 1, 'pa_only'),
    'time_only': (1, 1, 0, 0, 'time_only'),
    'pa_iq_only': (0, 0, 1, 0, None),
    'spectrum_only': (0, 0, 0, 1, None),
    'no_path': (1, 0, 1, 1, 'no_path_no_frequency'),
    'no_path_no_spectrum': (1, 0, 1, 0, None),
    'no_path_no_pa_iq': (1, 0, 0, 1, None),
    'time_without_path': (1, 0, 0, 0, 'no_path_time_only'),
}
STRUCTURES = ('single_view', 'no_time_coordinate', 'moment', 'shuffled_time')


def limits(gpu):
    if gpu not in range(8):
        raise ValueError('GPU outside registered range')
    return 1 if gpu == 7 else 3


def rows():
    out = []
    def add(setting, family, method, structure, seed, *, reuse=None, permuted=False):
        pid, display, _ = PROTOCOLS[setting]
        supervised = method.startswith('sup') or method == 'ce'
        aug = family != 'architecture' and method not in ('sup_noaug', 'pl_noaug')
        row_id = f'{setting}-{family}-{method}-{structure}-s{seed}'
        out.append(dict(row_id=row_id, data_setting=setting, protocol_id=pid, split_name=display,
            family=family, method=method, structure=structure, permuted=permuted,
            signature_views=1 if structure == 'single_view' else (0 if family == 'branch' and not BRANCHES[structure][1] else 3),
            channel_augmentation=aug, learning_regime='supervised' if supervised else 'ssl' if method == 'ssl' else 'pl',
            model_seed=seed, epochs=200 if family == 'architecture' else 50,
            updates=10000 if family == 'architecture' else 11100,
            views=['clean'] if family == 'architecture' else list(VIEWS),
            readouts=list(READOUTS) if family == 'factorial' else ['raw'],
            reuse_key=reuse, new_training=reuse is None))
    for structure, (_, _, _, _, previous) in BRANCHES.items():
        for seed in SEEDS:
            add('main_split', 'branch', 'ce', structure, seed,
                reuse=f'{previous}-s{seed}' if previous else None)
    for structure in STRUCTURES:
        for seed in SEEDS:
            add('main_split', 'structure_ce', 'ce', structure, seed, permuted=structure == 'shuffled_time')
    for setting in PROTOCOLS:
        for method in ('sup', 'pl', 'sup_noaug', 'pl_noaug', 'ssl'):
            for seed in SEEDS:
                add(setting, 'factorial', method, 'full', seed, permuted=method == 'ssl')
    for seed in EXTENDED_SEEDS[3:]:
        add('main_split', 'factorial', 'ssl', 'full', seed, permuted=True)
    for seed in SEEDS:
        add('main_split', 'factorial', 'pl_permuted', 'full', seed, permuted=True)
    for seed in EXTENDED_SEEDS:
        add('main_split', 'architecture', 'ce', 'full', seed)
    assert len(out) == 226 and len({r['row_id'] for r in out}) == 226
    assert sum(r['reuse_key'] is not None for r in out) == 15
    return out


def configuration(context, row, project, commit):
    if row not in rows() or not row['new_training']:
        raise ValueError('Registered fresh row required')
    if row['family'] in ('branch', 'structure_ce'):
        c = deepcopy(context['baselines'][str(row['model_seed'])]['complete']['config'])
        c.update(run_id=RUN, row_id=row['row_id'], code_commit=commit,
                 output_root=project+'/runs/'+RUN+'/F/'+row['row_id']+'/source',
                 algorithm_id='pamf_split_spectrum_structure_ce', paper_method='PAMF CE structural control',
                 paper_row=deepcopy(row), shared_prefix=None)
        c['repl_row'].update(row_id=row['row_id'], method='SS_'+row['structure'],
                             structure='SS_'+row['structure'], reuse=False, readouts=['raw'])
        return validate(c)
    from experiments.pamf_paper_completion import plan as old
    ctx = deepcopy(context)
    seed = str(row['model_seed'])
    if seed not in ctx['templates']:
        ctx['templates'][seed] = deepcopy(ctx['templates'][str(SEEDS[0])])
    saved = old.RUN, old.make_plan, old.CE_VARIANT
    old.RUN, old.make_plan, old.CE_VARIANT = RUN, lambda: dict(rows=rows()), CE_VARIANT
    try:
        c = old.configuration(ctx, row, project, commit)
    finally:
        old.RUN, old.make_plan, old.CE_VARIANT = saved
    if row['family'] != 'architecture':
        c['channel_augmentation'] = row['channel_augmentation']
    return validate(c)


def validate(c):
    row = c.get('paper_row')
    expected = next((r for r in rows() if r['row_id'] == c.get('row_id')), None)
    if (row != expected or expected is None or not row['new_training'] or
            c.get('run_id') != RUN or c.get('model_seed') != row['model_seed']):
        raise ValueError('Registered row identity required')
    if any(c.get(k) for k in ('truth', 'target_truth', 'views_root', 'tests', 'target_inputs',
                               'checkpoint', 'teacher_checkpoint', 'shared_prefix')):
        raise ValueError('Only source inputs permitted')
    if row['family'] == 'architecture':
        if (c.get('variant') != CE_VARIANT or c.get('epochs') != 200 or c.get('augmentation') or
                c.get('extra_losses') or c.get('selection') != 'fixed_last_epoch'):
            raise ValueError('Fixed E200 pure CE required')
    elif (c.get('epochs') != 50 or c.get('optimizer_updates') != 11100 or
          c.get('num_classes') != (10 if row['data_setting'] == 'ten_class' else 6) or
          c.get('own_EMA') or c.get('native_U_gate') or c.get('posthoc_head_fit')):
        raise ValueError('Fixed E50 source contract differs')
    elif row['family'] in ('branch', 'structure_ce'):
        if c.get('U_use') or c.get('ablation') != 'F15' or c.get('satellite_CE_weight') != .68:
            raise ValueError('CE-only branch recipe differs')
    elif c.get('U_use') != (row['learning_regime'] != 'supervised') or c.get('channel_augmentation') != row['channel_augmentation']:
        raise ValueError('Main method or augmentation differs')
    return c
