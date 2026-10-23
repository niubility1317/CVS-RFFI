"""Source-fixed same-predicted-TX cross-RX U moment mixing; no inference change."""
from copy import deepcopy
from .plan import SEEDS, validate as validate_scaffold
from .view_anchor import PARENT_RUN, PARENT_ALGORITHM, RESOURCE_GPU_BY_SEED
from .paired_u import RUN as PAIR_RUN
from .u_cric import RUN as UCRIC_RUN

RUN='20261023-phase1-PAMF-U-MixStyle-m3-r01'
ALGORITHM='pamf_u_mixstyle_v1'
RECIPE=dict(start_epoch=31,levels=['time_down','t2.pool'],probability_per_level=.5,alpha=.1,eps=1e-6,
    eligible_weight_min=.5,same_teacher_argmax=True,different_original_source_RX=True,
    random_uniform_eligible_partner=True,invalid_pair_identity=True,detached_moments=True,
    shared_partner_per_U_forward=True,shared_lambda_per_U_forward=True,
    teacher_and_L_hooks=False,new_student_forwards=0,new_inference_parameters=0,G_training_reads=0)


def configuration(parent,project,commit):
    c=deepcopy(parent);seed=c['model_seed']
    if (seed not in SEEDS or c.get('run_id')!=PARENT_RUN or c.get('row_id')!='RXPSM-s'+str(seed)
        or c.get('algorithm_id')!=PARENT_ALGORITHM or c.get('ssl_method')!='RXPSM'
        or any(c.get(k) for k in ('view_anchor','paired_u','u_cric','u_mixstyle'))):
        raise ValueError('Fixed original same-seed RXPSM E30 source required')
    c.update(run_id=RUN,row_id='UMIX-s'+str(seed),algorithm_id=ALGORITHM,
        display_method='PAMF same-predicted-TX cross-RX unlabeled moment mixing',
        output_root=project+'/runs/'+RUN+'/UMIX-s'+str(seed)+'/source',
        fork_parent_source=project+'/runs/'+PARENT_RUN+'/RXPSM-s'+str(seed)+'/source',
        u_mixstyle=deepcopy(RECIPE),code_commit=commit)
    return validate(c)


def validate(c):
    validate_scaffold(c)
    if (c.get('run_id')!=RUN or c.get('row_id')!='UMIX-s'+str(c['model_seed'])
        or c.get('algorithm_id')!=ALGORITHM or c.get('ssl_method')!='RXPSM' or c.get('u_mixstyle')!=RECIPE
        or any(c.get(k) for k in ('view_anchor','paired_u','u_cric','source_restorer'))
        or not c.get('fork_parent_source','').endswith('/runs/'+PARENT_RUN+'/RXPSM-s'+str(c['model_seed'])+'/source')):
        raise ValueError('Fixed U-MixStyle recipe and same-seed original E30 source required')
    return c


def second_slot_ready(rt,seed):
    from experiments.cric_dura_comparison.io import read
    paired=read(rt['paired_GPU_owner_state'])['jobs'].get('PAIREDU-s'+str(seed),{})
    ucric=read(rt['gpu_predecessor_state'])['jobs'].get('UCRIC-s'+str(seed),{})
    return (paired.get('status') in ('PREDICTIONS_COMPLETE','SCORING','SCORED')
        and ucric.get('status') in ('RUNNING','PREDICTIONS_COMPLETE','SCORING','SCORED'))
