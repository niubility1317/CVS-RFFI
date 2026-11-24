"""Fixed matched screen on the six-class WiSig Main split."""

from copy import deepcopy
from pathlib import Path

from experiments.pamf_repl_view_shuffle.plan import RUN as ORIGINAL_RUN, TASKS, VIEWS
from experiments.pamf_repl_view_shuffle.plan import validate_config as validate_original

RUN = '20261124-phase1-pamf-wisig-branch-style-m10-r01'
SEED = 2026092701
ORIGINAL_ROW = f'main_split-repl-shuffled_time-3view-s{SEED}'
ARMS = {
    'time': (True, ()),
    'none': (False, ()),
    'frequency-moments': (False, ('frequency_moments',)),
    'frequency-envelope': (False, ('frequency_envelope',)),
    'pa-moments': (False, ('pa_moments',)),
    'pa-envelope': (False, ('pa_envelope',)),
    'reference-trend': (False, ('reference_trend',)),
    'signature-bend': (False, ('signature_bend',)),
    'time-frequency-envelope': (True, ('frequency_envelope',)),
    'all-structured': (True, ('frequency_envelope', 'pa_envelope', 'reference_trend', 'signature_bend')),
}


def row_id(arm):
    return f'{arm}-s{SEED}'


def source_config(runtime, context, arm):
    original = deepcopy(context['original_source_config'])
    validate_original(original)
    c = deepcopy(original)
    c.update(run_id=RUN, row_id=row_id(arm), code_commit=runtime['code_commit'],
             output_root=str(Path(runtime['run_root']) / 'F' / row_id(arm) / 'source'),
             branch_style_arm=arm, shared_prefix=context['prefix'],
             repl_row=dict(original['repl_row'], row_id=row_id(arm)))
    validate_config(c)
    return c


def validate_config(c):
    arm = c.get('branch_style_arm')
    if arm not in ARMS or c['run_id'] != RUN or c['row_id'] != row_id(arm) or c['model_seed'] != SEED:
        raise ValueError('Registered Main split branch-style row required')
    if any(c.get(key) for key in ('truth', 'target_truth', 'tests', 'views_root',
                                  'dataset_path', 'checkpoint', 'teacher_checkpoint', 'parent_source')):
        raise ValueError('Only the registered source prefix and source cache may enter training')
    before = dict(c, run_id=ORIGINAL_RUN, row_id=ORIGINAL_ROW, shared_prefix=None)
    validate_original(before)
    prefix = Path(c['shared_prefix'])
    if prefix.parts[-5:] != (ORIGINAL_RUN, 'F', ORIGINAL_ROW, 'source', 'segment20'):
        raise ValueError('Unregistered source ancestor')
    return c


def check_prefix(c):
    from experiments.pamf_repl_completion.source import parent

    prefix = parent(c, c['shared_prefix'])
    p = prefix['config']
    for key in ('view_comparison', 'source_cache', 'source_contract', 'plain_PL_threshold',
                'plain_PL_weight', 'unlabeled_batch', 'U_start_epoch', 'trajectory', 'repl_row'):
        value = c[key]
        if key == 'repl_row':
            value = dict(value, row_id=ORIGINAL_ROW)
        if p[key] != value:
            raise ValueError('Source prefix recipe differs: ' + key)
    if (p['run_id'] != ORIGINAL_RUN or p['row_id'] != ORIGINAL_ROW or prefix['epoch'] != 20
            or p.get('parent_source') or prefix.get('ancestors') != []
            or prefix['initialization'] != 'scratch' or p.get('step_offset') != 0):
        raise ValueError('E20 source-only prefix provenance differs')
    return prefix
