"""Source-view-gradient-derived, fixed original RXPSM E30+20."""
from copy import deepcopy
from .plan import SEEDS,validate as validate_scaffold
from .view_anchor import PARENT_RUN,PARENT_ALGORITHM,RESOURCE_GPU_BY_SEED
from .u_cmr import RUN as UCMR_RUN
from .u_dualanchor import RUN as UDAP_RUN

RUN='20261023-phase1-PAMF-U-ViewBalance-m3-r01'
ALGORITHM='pamf_u_viewbalance_v1'
RECIPE=dict(start_epoch=31,parameter_groups=['t2','cls_head','other'],
 t2_name_contains='.t2.',classifier_name_contains='.cls_head.',
 anchors=['original_complete_T1_L','mean_true_label_CE_of_existing_two_raw_L_forwards'],
 views=['ordinary_noise','physical_practical'],exact_original_half_mask=True,full_U_batch_denominator=True,
 original_L_and_weighted_U_scalar_losses_unchanged=True,project_each_view_before_combination=True,
 U_projection='closest_Euclidean_vector_with_nonnegative_dot_to_both_anchors',
 solver='enumerate_none_first_second_both_active_and_zero_choose_closest_feasible',
 both_active_Gram_pseudoinverse_rtol=1e-12,normalized_feasibility_tolerance=1e-10,
 anchor_squared_norm_floor=1e-12,tiny_either_anchor_drop_U=True,
 projected_view_squared_norm_floor=1e-12,nonzero_view_norm_to_full_L=.5,
 missing_view_budget_not_redistributed=True,normalization_can_increase_or_decrease_view_magnitude=True,
 numerical_infeasible_after_cast_drop_view=True,extra_student_forwards=0,
 new_inference_parameters=0,G_training_reads=0,no_higher_order_gradient=True)


def configuration(parent,project,commit):
    c=deepcopy(parent);seed=c['model_seed']
    forbidden=('view_anchor','paired_u','u_cric','u_mixstyle','u_gradprotect','u_cmr','u_dualanchor','u_viewbalance')
    if (seed not in SEEDS or c.get('run_id')!=PARENT_RUN or c.get('row_id')!='RXPSM-s'+str(seed)
        or c.get('algorithm_id')!=PARENT_ALGORITHM or c.get('ssl_method')!='RXPSM'
        or any(c.get(key) for key in forbidden)):
        raise ValueError('Original same-seed RXPSM E30 required for view balance')
    c.update(run_id=RUN,row_id='UVB-s'+str(seed),algorithm_id=ALGORITHM,
        display_method='PAMF view-balanced dual-protected unlabeled gradients',
        output_root=project+'/runs/'+RUN+'/UVB-s'+str(seed)+'/source',
        fork_parent_source=project+'/runs/'+PARENT_RUN+'/RXPSM-s'+str(seed)+'/source',
        u_viewbalance=deepcopy(RECIPE),code_commit=commit)
    return validate(c)


def validate(c):
    validate_scaffold(c)
    if (c.get('run_id')!=RUN or c.get('row_id')!='UVB-s'+str(c['model_seed'])
        or c.get('algorithm_id')!=ALGORITHM or c.get('ssl_method')!='RXPSM' or c.get('u_viewbalance')!=RECIPE
        or any(c.get(key) for key in ('view_anchor','paired_u','u_cric','u_mixstyle','u_gradprotect','u_cmr','u_dualanchor','source_restorer'))
        or not c.get('fork_parent_source','').endswith('/runs/'+PARENT_RUN+'/RXPSM-s'+str(c['model_seed'])+'/source')):
        raise ValueError('Fixed view balance recipe and original E30 source required')
    return c


def second_slot_ready(rt,seed):
    from experiments.cric_dura_comparison.io import read
    prior=read(rt['cmr_GPU_owner_state'])['jobs'].get('UCMR-s'+str(seed),{})
    other=read(rt['gpu_predecessor_state'])['jobs'].get('UDAP-s'+str(seed),{})
    return (prior.get('status') in ('PREDICTIONS_COMPLETE','SCORING','SCORED')
            and other.get('status') in ('RUNNING','PREDICTIONS_COMPLETE','SCORING','SCORED'))
