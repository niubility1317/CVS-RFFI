"""Source-defined follow-up: align paired physical U features after fixed RXPSM E30."""
from copy import deepcopy

from .plan import SEEDS, validate as validate_scaffold

RUN = '20261023-phase1-PAMF-viewanchor-m3-r01'
PARENT_RUN = '20261023-phase1-PAMF-physical-softmatch-m6-r02'
ALGORITHM = 'pamf_rxpsm_viewanchor_v1'
PARENT_ALGORITHM = 'pamf_rxpsm_v1'
ANCHOR = dict(weight=.10, start_epoch=31, physical_only=True, teacher_detached=True)


def configuration(parent_config, project, commit):
    c = deepcopy(parent_config)
    seed = c['model_seed']
    if (seed not in SEEDS or c.get('run_id') != PARENT_RUN
            or c.get('row_id') != 'RXPSM-s' + str(seed)
            or c.get('algorithm_id') != PARENT_ALGORITHM or c.get('ssl_method') != 'RXPSM'
            or c.get('view_anchor')):
        raise ValueError('Fixed source-only RXPSM parent required')
    parent = project + '/runs/' + PARENT_RUN + '/RXPSM-s' + str(seed) + '/source'
    c.update(run_id=RUN, row_id='VIEWANCHOR-s' + str(seed), algorithm_id=ALGORITHM,
             display_method='PAMF RXPSM + physical feature consistency',
             output_root=project + '/runs/' + RUN + '/VIEWANCHOR-s' + str(seed) + '/source',
             fork_parent_source=parent, view_anchor=deepcopy(ANCHOR), code_commit=commit)
    return validate(c)


def validate(c):
    validate_scaffold(c)
    if (c.get('run_id') != RUN or c.get('row_id') != 'VIEWANCHOR-s' + str(c['model_seed'])
            or c.get('algorithm_id') != ALGORITHM or c.get('ssl_method') != 'RXPSM'
            or c.get('view_anchor') != ANCHOR
            or not c.get('fork_parent_source', '').endswith(
                '/runs/' + PARENT_RUN + '/RXPSM-s' + str(c['model_seed']) + '/source')):
        raise ValueError('Fixed physical-feature follow-up and own-seed parent required')
    return c
