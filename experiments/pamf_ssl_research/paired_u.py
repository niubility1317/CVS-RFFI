"""Predeclared clean/physical U pair from the same RXPSM source E30 prefix."""
from copy import deepcopy
from .plan import SEEDS, validate as validate_scaffold
from .view_anchor import PARENT_RUN, PARENT_ALGORITHM, RESOURCE_GPU_BY_SEED

RUN = '20261023-phase1-PAMF-paired-U-m3-r01'
ALGORITHM = 'pamf_pairedu_v1'
RECIPE = dict(start_epoch=31, clean_CE=.50, physical_CE=.25,
              clean_prototype=.0125, physical_prototype=.0125, physical_feature=.10,
              all_U_paired=True, teacher_detached=True)
REFERENCE_RUN = '20261023-phase1-PAMF-SSL-DURA-reference-m9-r01'


def configuration(parent, project, commit):
    c = deepcopy(parent)
    seed = c['model_seed']
    if (seed not in SEEDS or c.get('run_id') != PARENT_RUN
            or c.get('row_id') != 'RXPSM-s' + str(seed)
            or c.get('algorithm_id') != PARENT_ALGORITHM or c.get('ssl_method') != 'RXPSM'
            or c.get('view_anchor') or c.get('paired_u')):
        raise ValueError('Fixed original same-seed RXPSM E30 source required')
    c.update(run_id=RUN, row_id='PAIREDU-s' + str(seed), algorithm_id=ALGORITHM,
             display_method='PAMF paired clean and physical unlabeled views',
             output_root=project + '/runs/' + RUN + '/PAIREDU-s' + str(seed) + '/source',
             fork_parent_source=project + '/runs/' + PARENT_RUN + '/RXPSM-s' + str(seed) + '/source',
             paired_u=deepcopy(RECIPE), code_commit=commit)
    return validate(c)


def validate(c):
    validate_scaffold(c)
    if (c.get('run_id') != RUN or c.get('row_id') != 'PAIREDU-s' + str(c['model_seed'])
            or c.get('algorithm_id') != ALGORITHM or c.get('ssl_method') != 'RXPSM'
            or c.get('paired_u') != RECIPE or c.get('view_anchor')
            or not c.get('fork_parent_source', '').endswith(
                '/runs/' + PARENT_RUN + '/RXPSM-s' + str(c['model_seed']) + '/source')):
        raise ValueError('Fixed paired-U recipe and same-seed original E30 source required')
    return c


def reference_gpu_work_finished(rt, seed):
    """Per-card handoff: the previous owner has no more GPU rows for this seed."""
    from experiments.cric_dura_comparison.io import read
    state = read(rt['gpu_predecessor_state'])
    expected = {m + '-s' + str(seed) for m in ('PSM', 'RXPSM', 'VIEWANCHOR')}
    return all(state['jobs'].get(k, {}).get('status') in
               ('PREDICTIONS_COMPLETE', 'SCORING', 'SCORED') for k in expected)
