"""Only remove the raw U/L magnitude cap from the complete actual-step guard."""
from copy import deepcopy
from .hard_pl_step_guard import configuration as base_configuration,validate as base_validate,RUN as SG_RUN,METHODS as SG_METHODS
from .hard_pl_branch_guard import RUN as BG_RUN,METHODS as BG_METHODS,PROJECTION,ANCHOR
from .hard_pl_native_cap2 import CAP2
from .hard_pl import SEEDS,RESOURCE_GPU_BY_SEED

RUN='20261024-phase1-PAMF-HardPL-StepNoCap-m3-r01'
METHODS=('HPL_E20_AS_SGNC',)
NO_CAP=dict(raw_U_to_L_norm_cap=None,raw_direction_anchors=['L_full','L_clean','L_As'],raw_norm_nonexpansion=True,
    actual_parameter_guard='unchanged_StepGuard',gradient_calls=4,extra_teacher_student_head_forwards=0,new_head=False)


def configuration(template,project,commit):
    c=base_configuration(template,project,commit);seed=c['model_seed']
    c.pop('hard_pl_native_cap2')
    c['hard_pl_gradient_guard']['U_to_L_norm_cap']=None;c['hard_pl_branch_guard']['U_to_L_norm_cap']=None
    c.update(run_id=RUN,row_id=METHODS[0]+'-s'+str(seed),algorithm_id='pamf_hard_pl_step_no_raw_cap_v1',
        display_method='PAMF 高权重伪标签与无幅度帽更新保护',hard_pl_step_no_cap=deepcopy(NO_CAP),
        output_root=project+'/runs/'+RUN+'/'+METHODS[0]+'-s'+str(seed)+'/source')
    return validate(c)


def validate(c):
    seed=c.get('model_seed')
    if (c.get('run_id')!=RUN or c.get('row_id')!=METHODS[0]+'-s'+str(seed)
            or c.get('algorithm_id')!='pamf_hard_pl_step_no_raw_cap_v1' or c.get('hard_pl_step_no_cap')!=NO_CAP
            or 'hard_pl_native_cap2' in c or c.get('hard_pl_gradient_guard',{}).get('U_to_L_norm_cap','missing') is not None
            or c.get('hard_pl_branch_guard',{}).get('U_to_L_norm_cap','missing') is not None):
        raise ValueError('Only explicit uncapped raw magnitude with unchanged raw/actual-step protections allowed')
    proxy=deepcopy(c);proxy.pop('hard_pl_step_no_cap')
    proxy['hard_pl_gradient_guard']['U_to_L_norm_cap']=2.;proxy['hard_pl_branch_guard']['U_to_L_norm_cap']=2.
    proxy['hard_pl_native_cap2']=dict(CAP2,projection=PROJECTION,anchor=ANCHOR,extra_autograd_passes=2)
    proxy.update(run_id=SG_RUN,row_id=SG_METHODS[0]+'-s'+str(seed),algorithm_id='pamf_hard_pl_parameter_step_guard_v1')
    base_validate(proxy);return c


def slot_ready(rt,seed):
    from experiments.cric_dura_comparison.io import read
    done=('PREDICTIONS_COMPLETE','SCORING','SCORED')
    old=read(rt['branch_guard_GPU_owner_state'])['jobs'].get(BG_METHODS[0]+'-s'+str(seed),{})
    peer=read(rt['step_guard_GPU_owner_state'])['jobs'].get(SG_METHODS[0]+'-s'+str(seed),{})
    return old.get('status') in done and peer.get('status') in ('RUNNING',)+done


def reserved_predecessor_pids(rt,seed):
    from experiments.cric_dura_comparison.io import read
    result=[]
    for path,method in ((rt['branch_guard_GPU_owner_state'],BG_METHODS[0]),(rt['step_guard_GPU_owner_state'],SG_METHODS[0])):
        job=read(path)['jobs'].get(method+'-s'+str(seed),{})
        if job.get('status')=='RUNNING':result.append(int(job['pid']))
    return tuple(result)


def smoke_slot(rt,capacity):
    for seed in SEEDS:
        if slot_ready(rt,seed):
            gpu=RESOURCE_GPU_BY_SEED[seed];apps,free=capacity(gpu,reserved_predecessor_pids(rt,seed))
            if apps<3 and free>=10000:return seed,gpu
    return None
