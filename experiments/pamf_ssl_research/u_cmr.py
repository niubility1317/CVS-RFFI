"""Fixed raw-U conditional receiver moments, independent of current tested variants."""
from copy import deepcopy
from .plan import SEEDS,validate as validate_scaffold
from .view_anchor import PARENT_RUN,PARENT_ALGORITHM,RESOURCE_GPU_BY_SEED
from .u_mixstyle import RUN as UMIX_RUN
from .u_gradprotect import RUN as UGP_RUN

RUN='20261023-phase1-PAMF-U-CMR-m3-r01'
ALGORITHM='pamf_u_cmr_v1'
RECIPE=dict(start_epoch=31,weight=.1,levels=['time_down','t2.pool'],eligible_weight_min=.5,
 min_effective_samples=4.,descriptor_std_floor=1e-4,moment_eps=1e-6,
 descriptor='temporal_mean_and_population_std',detached_global_batch_standardization=True,
 weighted_population_mean_std=True,effective_sample_size='sum_w_squared_over_sum_squared_w',
 pairing='same_existing_target_argmax_different_original_source_RX',
 pair_penalty='mean_over_dimensions((mean_r-mean_s)^2+(std_r-std_s)^2)',
 aggregation='mean_pairs_per_valid_class_then_mean_classes_then_mean_levels',
 targets_and_confidence_detached=True,raw_U_student=True,extra_student_forwards=1,
 teacher_and_L_unchanged=True,new_inference_parameters=0,moment_EMA_bank=False,G_training_reads=0)


def configuration(parent,project,commit):
    c=deepcopy(parent);seed=c['model_seed']
    if (seed not in SEEDS or c.get('run_id')!=PARENT_RUN or c.get('row_id')!='RXPSM-s'+str(seed)
        or c.get('algorithm_id')!=PARENT_ALGORITHM or c.get('ssl_method')!='RXPSM'
        or any(c.get(k) for k in ('view_anchor','paired_u','u_cric','u_mixstyle','u_gradprotect','u_cmr'))):
        raise ValueError('Fixed original same-seed RXPSM E30 required')
    c.update(run_id=RUN,row_id='UCMR-s'+str(seed),algorithm_id=ALGORITHM,
        display_method='PAMF raw-U pseudo-class conditional receiver moment matching',
        output_root=project+'/runs/'+RUN+'/UCMR-s'+str(seed)+'/source',
        fork_parent_source=project+'/runs/'+PARENT_RUN+'/RXPSM-s'+str(seed)+'/source',
        u_cmr=deepcopy(RECIPE),code_commit=commit)
    return validate(c)


def validate(c):
    validate_scaffold(c)
    if (c.get('run_id')!=RUN or c.get('row_id')!='UCMR-s'+str(c['model_seed'])
        or c.get('algorithm_id')!=ALGORITHM or c.get('ssl_method')!='RXPSM' or c.get('u_cmr')!=RECIPE
        or any(c.get(k) for k in ('view_anchor','paired_u','u_cric','u_mixstyle','u_gradprotect','source_restorer'))
        or not c.get('fork_parent_source','').endswith('/runs/'+PARENT_RUN+'/RXPSM-s'+str(c['model_seed'])+'/source')):
        raise ValueError('Fixed U-CMR recipe and original E30 source required')
    return c


def second_slot_ready(rt,seed):
    from experiments.cric_dura_comparison.io import read
    prior=read(rt['umix_GPU_owner_state'])['jobs'].get('UMIX-s'+str(seed),{})
    other=read(rt['gpu_predecessor_state'])['jobs'].get('UGP-s'+str(seed),{})
    return (prior.get('status') in ('PREDICTIONS_COMPLETE','SCORING','SCORED')
            and other.get('status') in ('RUNNING','PREDICTIONS_COMPLETE','SCORING','SCORED'))
