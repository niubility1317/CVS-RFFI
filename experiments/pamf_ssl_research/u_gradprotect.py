"""Source-gradient-derived fixed E30+20, with L-anchored U gradient combination."""
from copy import deepcopy
from .plan import SEEDS,validate as validate_scaffold
from .view_anchor import PARENT_RUN,PARENT_ALGORITHM,RESOURCE_GPU_BY_SEED
from .u_cric import RUN as UCRIC_RUN
from .u_mixstyle import RUN as UMIX_RUN

RUN='20261023-phase1-PAMF-U-GradProtect-m3-r01'
ALGORITHM='pamf_u_gradprotect_v1'
RECIPE=dict(start_epoch=31,parameter_groups=['t2','cls_head','other'],
 t2_name_contains='.t2.',classifier_name_contains='.cls_head.',one_sided_U_projection=True,
 protect_original_T1_L=True,weighted_U_loss_unchanged=True,projection_if_dot_negative=True,
 norm_cap_U_to_L=1.,L_squared_norm_floor=1e-12,zero_or_tiny_L_group_drop_U=True,
 cap_after_projection=True,extra_student_forwards=0,new_inference_parameters=0,G_training_reads=0,
 no_higher_order_gradient=True)


def configuration(parent,project,commit):
    c=deepcopy(parent);seed=c['model_seed']
    if (seed not in SEEDS or c.get('run_id')!=PARENT_RUN or c.get('row_id')!='RXPSM-s'+str(seed)
        or c.get('algorithm_id')!=PARENT_ALGORITHM or c.get('ssl_method')!='RXPSM'
        or any(c.get(k) for k in ('view_anchor','paired_u','u_cric','u_mixstyle','u_gradprotect'))):
        raise ValueError('Fixed original same-seed RXPSM E30 required')
    c.update(run_id=RUN,row_id='UGP-s'+str(seed),algorithm_id=ALGORITHM,
        display_method='PAMF source-L anchored unlabeled gradient protection',
        output_root=project+'/runs/'+RUN+'/UGP-s'+str(seed)+'/source',
        fork_parent_source=project+'/runs/'+PARENT_RUN+'/RXPSM-s'+str(seed)+'/source',
        u_gradprotect=deepcopy(RECIPE),code_commit=commit)
    return validate(c)


def validate(c):
    validate_scaffold(c)
    if (c.get('run_id')!=RUN or c.get('row_id')!='UGP-s'+str(c['model_seed'])
        or c.get('algorithm_id')!=ALGORITHM or c.get('ssl_method')!='RXPSM'
        or c.get('u_gradprotect')!=RECIPE
        or any(c.get(k) for k in ('view_anchor','paired_u','u_cric','u_mixstyle','source_restorer'))
        or not c.get('fork_parent_source','').endswith('/runs/'+PARENT_RUN+'/RXPSM-s'+str(c['model_seed'])+'/source')):
        raise ValueError('Fixed source-gradient protection and original E30 source required')
    return c


def second_slot_ready(rt,seed):
    from experiments.cric_dura_comparison.io import read
    prior=read(rt['ucric_GPU_owner_state'])['jobs'].get('UCRIC-s'+str(seed),{})
    other=read(rt['gpu_predecessor_state'])['jobs'].get('UMIX-s'+str(seed),{})
    return (prior.get('status') in ('PREDICTIONS_COMPLETE','SCORING','SCORED')
            and other.get('status') in ('RUNNING','PREDICTIONS_COMPLETE','SCORING','SCORED'))
