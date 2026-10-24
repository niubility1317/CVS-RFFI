"""One source-fixed three-anchor U operator under the complete nativeAs cap2 recipe."""
from copy import deepcopy
from .hard_pl_native_cap2 import configuration as base_configuration,validate as base_validate,RUN as CAP_RUN,METHODS as CAP_METHODS,CAP2
from .hard_pl_balanced_ce import METHODS as BCE_METHODS
from .hard_pl import SEEDS,RESOURCE_GPU_BY_SEED

RUN='20261024-phase1-PAMF-HardPL-BranchGuard-m3-r01'
METHODS=('HPL_E20_AS_BG2',)
PROJECTION='simultaneous_three_L_anchors_per_group'
ANCHOR='complete_L_nonphysical_L_physical_L'
BRANCH_GUARD=dict(anchors=['L_full','L_clean','L_As'],projection=PROJECTION,U_to_L_norm_cap=2.,
    components='same_four_original_L_pair_results',nonphysical_coefficients=[.5,.25,.25],physical_coefficient=.25,
    full_L_unchanged=True,solver='thin_QR_and_active_set_SVD',gradient_calls=4,extra_autograd_passes=2,extra_teacher_or_student_forward=0,
    raw_gate=.95,U_weight=6.,tiny_L_squared_norm=1e-12,pinv_rtol=1e-10,KKT_relative_tolerance=1e-8,
    actual_cast_constraint_cosine_tolerance=1e-6,new_head=False)


def configuration(template,project,commit):
    c=base_configuration(template,project,commit);seed=c['model_seed']
    c['hard_pl_gradient_guard'].update(projection=PROJECTION,anchor=ANCHOR)
    c['hard_pl_native_cap2'].update(projection=PROJECTION,anchor=ANCHOR,extra_autograd_passes=2)
    c.update(run_id=RUN,row_id=METHODS[0]+'-s'+str(seed),algorithm_id='pamf_hard_pl_branch_guard_v1',
        display_method='PAMF 高权重伪标签与监督分支同时保护',hard_pl_branch_guard=deepcopy(BRANCH_GUARD),
        output_root=project+'/runs/'+RUN+'/'+METHODS[0]+'-s'+str(seed)+'/source')
    return validate(c)


def validate(c):
    seed=c.get('model_seed');guard=c.get('hard_pl_gradient_guard',{})
    if (c.get('run_id')!=RUN or c.get('row_id')!=METHODS[0]+'-s'+str(seed)
            or c.get('algorithm_id')!='pamf_hard_pl_branch_guard_v1' or c.get('hard_pl_branch_guard')!=BRANCH_GUARD
            or c.get('hard_pl_native_cap2')!=dict(CAP2,projection=PROJECTION,anchor=ANCHOR,extra_autograd_passes=2)
            or guard.get('projection')!=PROJECTION or guard.get('anchor')!=ANCHOR):
        raise ValueError('Fixed nativeAs cap2 with the complete and both original L branch constraints required')
    proxy=deepcopy(c);proxy.pop('hard_pl_branch_guard');proxy['hard_pl_native_cap2']=deepcopy(CAP2)
    proxy['hard_pl_gradient_guard'].update(projection='one_sided_per_group',anchor='complete_current_L_objective')
    proxy.update(run_id=CAP_RUN,row_id=CAP_METHODS[0]+'-s'+str(seed),algorithm_id='pamf_hard_pl_native_cap2_v1')
    base_validate(proxy);return c


def slot_ready(rt,seed):
    from experiments.cric_dura_comparison.io import read
    done=('PREDICTIONS_COMPLETE','SCORING','SCORED')
    old=read(rt['balanced_ce_GPU_owner_state'])['jobs'].get(BCE_METHODS[0]+'-s'+str(seed),{})
    peer=read(rt['native_cap2_GPU_owner_state'])['jobs'].get(CAP_METHODS[0]+'-s'+str(seed),{})
    return old.get('status') in done and peer.get('status') in ('RUNNING',)+done


def reserved_predecessor_pids(rt,seed):
    from experiments.cric_dura_comparison.io import read
    result=[]
    for path,method in ((rt['balanced_ce_GPU_owner_state'],BCE_METHODS[0]),(rt['native_cap2_GPU_owner_state'],CAP_METHODS[0])):
        job=read(path)['jobs'].get(method+'-s'+str(seed),{})
        if job.get('status')=='RUNNING':result.append(int(job['pid']))
    return tuple(result)


def smoke_slot(rt,capacity):
    for seed in SEEDS:
        if not slot_ready(rt,seed):continue
        gpu=RESOURCE_GPU_BY_SEED[seed];apps,free=capacity(gpu,reserved_predecessor_pids(rt,seed))
        if apps<3 and free>=10000:return seed,gpu
    return None
