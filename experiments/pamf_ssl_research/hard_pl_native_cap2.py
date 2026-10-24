"""Fixed full nativeAs objective with only its actual U/L gradient cap raised to two."""
from copy import deepcopy
from .hard_pl_native_as import configuration as base_configuration,validate as base_validate,RUN as NAS_RUN,METHODS as NAS_METHODS
from .hard_pl_balanced_ce import METHODS as BCE_METHODS
from .hard_pl import SEEDS,RESOURCE_GPU_BY_SEED

RUN='20261024-phase1-PAMF-HardPL-NativeAsCap2-m3-r01'
METHODS=('HPL_E20_AS_CAP2',)
CAP2=dict(U_to_L_norm_cap=2.,anchor='complete_current_L_objective',projection='one_sided_per_group',
    groups=['t2','cls_head','other'],tiny_L_squared_norm=1e-12,raw_gate=.95,U_weight=6.,
    native_As_policy_unchanged=True,loss_coefficients_unchanged=True,extra_autograd_passes=0,
    extra_teacher_or_student_forward=0,new_head=False)


def configuration(template,project,commit):
    c=base_configuration(template,project,commit);seed=c['model_seed']
    c['hard_pl_gradient_guard']['U_to_L_norm_cap']=2.
    c.update(run_id=RUN,row_id=METHODS[0]+'-s'+str(seed),algorithm_id='pamf_hard_pl_native_cap2_v1',
        display_method='PAMF 原生信道视图与双倍 U 范数帽',hard_pl_native_cap2=deepcopy(CAP2),
        output_root=project+'/runs/'+RUN+'/'+METHODS[0]+'-s'+str(seed)+'/source')
    return validate(c)


def validate(c):
    seed=c.get('model_seed')
    if (c.get('run_id')!=RUN or c.get('row_id')!=METHODS[0]+'-s'+str(seed)
        or c.get('algorithm_id')!='pamf_hard_pl_native_cap2_v1' or c.get('hard_pl_native_cap2')!=CAP2
        or c.get('hard_pl_gradient_guard',{}).get('U_to_L_norm_cap')!=2.):
        raise ValueError('Fixed nativeAs cap2,unchanged completeL projection and original loss recipe required')
    proxy=deepcopy(c);proxy.pop('hard_pl_native_cap2');proxy['hard_pl_gradient_guard']['U_to_L_norm_cap']=1.
    proxy.update(run_id=NAS_RUN,row_id=NAS_METHODS[0]+'-s'+str(seed),algorithm_id='pamf_hard_pl_native_as_v1')
    base_validate(proxy);return c


def slot_ready(rt,seed):
    from experiments.cric_dura_comparison.io import read
    done=('PREDICTIONS_COMPLETE','SCORING','SCORED')
    old=read(rt['native_as_GPU_owner_state'])['jobs'].get(NAS_METHODS[0]+'-s'+str(seed),{})
    peer=read(rt['balanced_ce_GPU_owner_state'])['jobs'].get(BCE_METHODS[0]+'-s'+str(seed),{})
    return old.get('status') in done and peer.get('status') in ('RUNNING',)+done


def reserved_predecessor_pids(rt,seed):
    from experiments.cric_dura_comparison.io import read
    result=[]
    for path,method in ((rt['native_as_GPU_owner_state'],NAS_METHODS[0]),(rt['balanced_ce_GPU_owner_state'],BCE_METHODS[0])):
        job=read(path)['jobs'].get(method+'-s'+str(seed),{})
        if job.get('status')=='RUNNING':result.append(int(job['pid']))
    return tuple(result)


def smoke_slot(rt,capacity):
    for seed in SEEDS:
        if not slot_ready(rt,seed):continue
        gpu=RESOURCE_GPU_BY_SEED[seed];apps,free=capacity(gpu,reserved_predecessor_pids(rt,seed))
        if apps<3 and free>=10000:return seed,gpu
    return None
