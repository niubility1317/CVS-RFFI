"""Source-supported actual-AdamW-step guard on the fixed complete nativeAs branch recipe."""
from copy import deepcopy
from .hard_pl_branch_guard import configuration as base_configuration,validate as base_validate,RUN as BG_RUN,METHODS as BG_METHODS
from .hard_pl import SEEDS,RESOURCE_GPU_BY_SEED

RUN='20261024-phase1-PAMF-HardPL-StepGuard-m3-r01'
METHODS=('HPL_E20_AS_SG2',)
STEP_GUARD=dict(reference='same_current_parameters_and_live_AdamW_moments_L_only',
    raw_step='original_joint_branch_guard_CENTER_UNITS_AdamW',
    anchors=['L_full','L_clean','L_As'],projection='actual_FP32_extra_descent_three_anchor_halfspaces',
    groups=['t2','cls_head','other'],solver='thin_QR_active_set_SVD_affine_quantization_repair',
    tiny_L_squared_norm=1e-12,component_omit_squared_norm=1e-12,pinv_rtol=1e-10,KKT_relative_tolerance=1e-8,
    actual_cast_constraint_cosine_tolerance=1e-6,norm_nonexpansion_relative=1e-6,norm_nonexpansion_absolute=1e-12,
    quantization_attempts=4,repair_margin='2_sum_abs_unit_anchor_times_rounding_error_plus_1e-12_max_rawnorm_1e-12',
    failed_quantization='exact_L_reference_group_with_explicit_counts',
    real_joint_moments_retained=True,reference_optimizer_steps_per_step=1,
    extra_autograd_calls=0,extra_model_forwards=0,new_head=False,new_inference_parameters=0)


def configuration(template,project,commit):
    c=base_configuration(template,project,commit);seed=c['model_seed']
    c.update(run_id=RUN,row_id=METHODS[0]+'-s'+str(seed),algorithm_id='pamf_hard_pl_parameter_step_guard_v1',
        display_method='PAMF 高权重伪标签与实际参数更新保护',hard_pl_step_guard=deepcopy(STEP_GUARD),
        output_root=project+'/runs/'+RUN+'/'+METHODS[0]+'-s'+str(seed)+'/source')
    return validate(c)


def validate(c):
    seed=c.get('model_seed')
    if (c.get('run_id')!=RUN or c.get('row_id')!=METHODS[0]+'-s'+str(seed)
            or c.get('algorithm_id')!='pamf_hard_pl_parameter_step_guard_v1' or c.get('hard_pl_step_guard')!=STEP_GUARD):
        raise ValueError('Fixed actual parameter increment guard with current-moment L reference required')
    proxy=deepcopy(c);proxy.pop('hard_pl_step_guard')
    proxy.update(run_id=BG_RUN,row_id=BG_METHODS[0]+'-s'+str(seed),algorithm_id='pamf_hard_pl_branch_guard_v1')
    base_validate(proxy);return c


def slot_ready(rt,seed):
    from experiments.cric_dura_comparison.io import read
    job=read(rt['branch_guard_GPU_owner_state'])['jobs'].get(BG_METHODS[0]+'-s'+str(seed),{})
    return job.get('status') in ('RUNNING','PREDICTIONS_COMPLETE','SCORING','SCORED')


def reserved_predecessor_pids(rt,seed):
    from experiments.cric_dura_comparison.io import read
    job=read(rt['branch_guard_GPU_owner_state'])['jobs'].get(BG_METHODS[0]+'-s'+str(seed),{})
    return (int(job['pid']),) if job.get('status')=='RUNNING' else ()


def smoke_slot(rt,capacity):
    for seed in SEEDS:
        if slot_ready(rt,seed):
            gpu=RESOURCE_GPU_BY_SEED[seed];apps,free=capacity(gpu,reserved_predecessor_pids(rt,seed))
            if apps<3 and free>=10000:return seed,gpu
    return None
