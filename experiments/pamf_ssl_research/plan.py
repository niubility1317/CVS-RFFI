"""Predeclared paired experiments; training does not consume target scores."""
from copy import deepcopy

RUN = '20261023-phase1-PAMF-physical-softmatch-m6-r01'
SEEDS = (2026092701, 2026092702, 2026092703)
METHODS = ('PSM', 'RXPSM')
GPU_BY_SEED = dict(zip(SEEDS, (0, 1, 2)))
VIEWS = ('clean', 'practical_high', 'practical_mid', 'practical_low_suburban',
         'practical_high_urban', 'practical_mid_urban', 'practical_low_urban')


def configuration(template, method, seed, project, commit):
    if method not in METHODS or seed not in SEEDS:
        raise ValueError('Undeclared method or model seed')
    c = deepcopy(template)
    for key in ('core_mechanisms', 'parent_source', 'step_offset', 'epoch_offset',
                'segment_epochs', 'replaces_source_row', 'F_parent_run'):
        c.pop(key, None)
    row = method + '-s' + str(seed)
    c.update(run_id=RUN, row_id=row, model_seed=seed, algorithm_id='pamf_' + method.lower() + '_v1',
             display_method='PAMF ' + method, ssl_method=method,
             # This is the existing trainer's scaffold selector, not the new algorithm name.
             method_id='pamf_t1_e50', scaffold_id='pamf_t1_e50_supervised', ablation='F21',
             regime='native_cric', satellite_concat=False, satellite_CE_weight=0.,
             native_U_gate=False, native_unlabeled_loop=False, own_EMA=True,
             U_use=True, U_start_epoch=14, unlabeled_batch=256,
             plain_PL_threshold=None, plain_PL_weight=None,
             ssl=dict(teacher_decay=.996, moment_decay=.999, confidence_floor=.70,
                      pseudo_temperature=.5, unsupervised_weight=.5, ramp_epochs=5,
                      physical_fraction=.5, prototype_temperature=.1,
                      prototype_momentum=.9, prototype_mix=.2,
                      prototype_feature_weight=.05, same_RX_prototypes=False,
                      unconditional_entropy_weight=0.),
             output_root=project + '/runs/' + RUN + '/' + row + '/source',
             code_commit=commit, initialization='same_contract_scratch20_own10_own20',
             default_head_joint=True, posthoc_head_fit=False)
    validate(c)
    return c


def validate(c):
    forbidden = ('truth', 'target_truth', 'target_inputs', 'views_root', 'tests',
                 'dataset_path', 'checkpoint', 'teacher_checkpoint')
    if any(c.get(k) for k in forbidden):
        raise ValueError('Training accepts source handles only')
    if (c.get('ssl_method') not in METHODS or c.get('model_seed') not in SEEDS
            or c.get('num_classes') != 6 or c.get('source_rxs') != [1, 3, 4, 6, 8]
            or c.get('source_days') != [1, 2, 3]
            or c.get('optimizer_updates') != 11100 or c.get('epochs') != 50
            or c.get('trajectory') != [20, 10, 20] or c.get('U_start_epoch') != 14
            or c.get('ablation') != 'F21' or c.get('satellite_concat') is not False
            or c.get('native_U_gate') is not False
            or c.get('ssl', {}).get('same_RX_prototypes') is not False):
        raise ValueError('Matched PAMF scaffold/data/budget required')
    return c
