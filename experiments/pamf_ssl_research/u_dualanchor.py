"""Source-raw-gradient-derived fixed E30+20 with two L anchors for U."""
from copy import deepcopy
from .plan import SEEDS,validate as validate_scaffold
from .view_anchor import PARENT_RUN,PARENT_ALGORITHM,RESOURCE_GPU_BY_SEED
from .u_gradprotect import RUN as UGP_RUN
from .u_cmr import RUN as UCMR_RUN

RUN='20261023-phase1-PAMF-U-DualAnchor-m3-r01'
ALGORITHM='pamf_u_dualanchor_v1'
RECIPE=dict(start_epoch=31,parameter_groups=['t2','cls_head','other'],
 t2_name_contains='.t2.',classifier_name_contains='.cls_head.',
 anchors=['original_complete_T1_L','mean_true_label_CE_of_existing_two_raw_L_forwards'],
 original_L_and_weighted_U_losses_unchanged=True,
 U_projection='closest_Euclidean_vector_with_nonnegative_dot_to_both_anchors',
 solver='enumerate_none_first_second_both_active_and_zero_choose_closest_feasible',
 both_active_Gram_pseudoinverse_rtol=1e-12,normalized_feasibility_tolerance=1e-10,
 anchor_squared_norm_floor=1e-12,tiny_either_anchor_drop_U=True,
 norm_cap_U_to_original_full_L=1.,cap_after_projection=True,
 numerical_infeasible_after_cast_drop_U=True,extra_student_forwards=0,
 new_inference_parameters=0,G_training_reads=0,no_higher_order_gradient=True)


def configuration(parent,project,commit):
    c=deepcopy(parent);seed=c['model_seed']
    if (seed not in SEEDS or c.get('run_id')!=PARENT_RUN or c.get('row_id')!='RXPSM-s'+str(seed)
        or c.get('algorithm_id')!=PARENT_ALGORITHM or c.get('ssl_method')!='RXPSM'
        or any(c.get(k) for k in ('view_anchor','paired_u','u_cric','u_mixstyle','u_gradprotect','u_cmr','u_dualanchor'))):
        raise ValueError('Original same-seed RXPSM E30 required for dual anchor')
    c.update(run_id=RUN,row_id='UDAP-s'+str(seed),algorithm_id=ALGORITHM,
        display_method='PAMF full-supervision and raw-identity dual anchored unlabeled gradients',
        output_root=project+'/runs/'+RUN+'/UDAP-s'+str(seed)+'/source',
        fork_parent_source=project+'/runs/'+PARENT_RUN+'/RXPSM-s'+str(seed)+'/source',
        u_dualanchor=deepcopy(RECIPE),code_commit=commit)
    return validate(c)


def validate(c):
    validate_scaffold(c)
    if (c.get('run_id')!=RUN or c.get('row_id')!='UDAP-s'+str(c['model_seed'])
        or c.get('algorithm_id')!=ALGORITHM or c.get('ssl_method')!='RXPSM'
        or c.get('u_dualanchor')!=RECIPE
        or any(c.get(k) for k in ('view_anchor','paired_u','u_cric','u_mixstyle','u_gradprotect','u_cmr','source_restorer'))
        or not c.get('fork_parent_source','').endswith('/runs/'+PARENT_RUN+'/RXPSM-s'+str(c['model_seed'])+'/source')):
        raise ValueError('Fixed dual anchor recipe and original same-seed E30 required')
    return c


def second_slot_ready(rt,seed):
    from experiments.cric_dura_comparison.io import read
    prior=read(rt['ugp_GPU_owner_state'])['jobs'].get('UGP-s'+str(seed),{})
    other=read(rt['gpu_predecessor_state'])['jobs'].get('UCMR-s'+str(seed),{})
    return (prior.get('status') in ('PREDICTIONS_COMPLETE','SCORING','SCORED')
            and other.get('status') in ('RUNNING','PREDICTIONS_COMPLETE','SCORING','SCORED'))
