"""Extend PAMF's existing shallow/middle cross-RX criterion to hidden-label U."""
from copy import deepcopy
from .plan import SEEDS, validate as validate_scaffold
from .view_anchor import PARENT_RUN, PARENT_ALGORITHM, RESOURCE_GPU_BY_SEED
from .paired_u import RUN as PAIR_RUN

RUN = '20261023-phase1-PAMF-U-CRIC-m3-r01'
ALGORITHM = 'pamf_u_cric_v1'
RECIPE = dict(start_epoch=31, weight=.10, temperature=.2, levels=['time_down', 't2.pool'],
              positives='same_teacher_argmax_other_source_RX', negatives='different_teacher_argmax',
              pair_soft_probability=True, anchor_and_neighbor_confidence=True, teacher_detached=True)


def configuration(parent, project, commit):
    c = deepcopy(parent)
    seed = c['model_seed']
    if (seed not in SEEDS or c.get('run_id') != PARENT_RUN
            or c.get('row_id') != 'RXPSM-s' + str(seed)
            or c.get('algorithm_id') != PARENT_ALGORITHM or c.get('ssl_method') != 'RXPSM'
            or any(c.get(k) for k in ('view_anchor', 'paired_u', 'u_cric'))):
        raise ValueError('Fixed original same-seed RXPSM E30 source required')
    c.update(run_id=RUN, row_id='UCRIC-s' + str(seed), algorithm_id=ALGORITHM,
             display_method='PAMF unlabeled shallow/middle cross-receiver contrast',
             output_root=project + '/runs/' + RUN + '/UCRIC-s' + str(seed) + '/source',
             fork_parent_source=project + '/runs/' + PARENT_RUN + '/RXPSM-s' + str(seed) + '/source',
             u_cric=deepcopy(RECIPE), code_commit=commit)
    return validate(c)


def validate(c):
    validate_scaffold(c)
    if (c.get('run_id') != RUN or c.get('row_id') != 'UCRIC-s' + str(c['model_seed'])
            or c.get('algorithm_id') != ALGORITHM or c.get('ssl_method') != 'RXPSM'
            or c.get('u_cric') != RECIPE or c.get('view_anchor') or c.get('paired_u')
            or not c.get('fork_parent_source', '').endswith(
                '/runs/' + PARENT_RUN + '/RXPSM-s' + str(c['model_seed']) + '/source')):
        raise ValueError('Fixed U-CRIC recipe and same-seed original E30 source required')
    return c


def paired_dispatch_settled(rt, seed):
    """A second card slot is safe only after the older owner dispatched its row."""
    from experiments.cric_dura_comparison.io import read
    state = read(rt['gpu_predecessor_state'])
    row = state['jobs'].get('PAIREDU-s' + str(seed), {})
    return row.get('status') in ('RUNNING', 'PREDICTIONS_COMPLETE', 'SCORING', 'SCORED')
