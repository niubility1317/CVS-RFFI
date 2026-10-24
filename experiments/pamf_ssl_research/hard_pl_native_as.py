"""Restore the original native source-As half under otherwise fixed Joint6G training."""
from copy import deepcopy
from .hard_pl_joint import configuration as base_configuration,validate as base_validate,RUN as JOINT_RUN,METHODS as JOINT_METHODS
from .hard_pl_path_reliable_as import METHODS as COMBO_METHODS
from .hard_pl_receiver_extrapolation import METHODS as RXE_METHODS
from .hard_pl import SEEDS,RESOURCE_GPU_BY_SEED

RUN='20261024-phase1-PAMF-HardPL-NativeAs-m3-r01'
METHODS=('HPL_E20_JOINT_AS6G',)
NATIVE_AS=dict(generator='same_native_source_Practical_as_supervised_L',physical_half='same_original_alternating_half',
    pseudo_labels='original_raw_argmax',raw_gate=.95,augmented_view_gate=None,raw_fallback=False,
    CE_and_KL_coefficients_unchanged=True,CE_normalizer=1.25,student_U_passes=6,
    extra_As_teacher=0,extra_student_forwards=0,new_head=False)


def configuration(template,project,commit):
    c=base_configuration(template,project,commit);seed=c['model_seed']
    c.update(run_id=RUN,row_id=METHODS[0]+'-s'+str(seed),algorithm_id='pamf_hard_pl_native_as_v1',
        display_method='PAMF 高权重伪标签原生信道视图',hard_pl_no_uas=False,hard_pl_native_as=deepcopy(NATIVE_AS),
        output_root=project+'/runs/'+RUN+'/'+METHODS[0]+'-s'+str(seed)+'/source')
    return validate(c)


def validate(c):
    seed=c.get('model_seed')
    if (c.get('run_id')!=RUN or c.get('row_id')!=METHODS[0]+'-s'+str(seed)
        or c.get('algorithm_id')!='pamf_hard_pl_native_as_v1' or c.get('hard_pl_no_uas') is not False
        or c.get('hard_pl_native_as')!=NATIVE_AS):
        raise ValueError('Fixed native sourceAs half,raw .95 gate and unchanged original Joint6G loss required')
    proxy=deepcopy(c);proxy.pop('hard_pl_native_as');proxy['hard_pl_no_uas']=True
    proxy.update(run_id=JOINT_RUN,row_id=JOINT_METHODS[0]+'-s'+str(seed),algorithm_id='pamf_hard_pl_joint160_v1')
    base_validate(proxy);return c


def slot_ready(rt,seed):
    from experiments.cric_dura_comparison.io import read
    done=('PREDICTIONS_COMPLETE','SCORING','SCORED')
    old=read(rt['path_reliable_as_GPU_owner_state'])['jobs'].get(COMBO_METHODS[0]+'-s'+str(seed),{})
    peer=read(rt['receiver_extrapolation_GPU_owner_state'])['jobs'].get(RXE_METHODS[0]+'-s'+str(seed),{})
    return old.get('status') in done and peer.get('status') in ('RUNNING',)+done


def reserved_predecessor_pids(rt,seed):
    from experiments.cric_dura_comparison.io import read
    result=[]
    for path,method in ((rt['path_reliable_as_GPU_owner_state'],COMBO_METHODS[0]),(rt['receiver_extrapolation_GPU_owner_state'],RXE_METHODS[0])):
        job=read(path)['jobs'].get(method+'-s'+str(seed),{})
        if job.get('status')=='RUNNING':result.append(int(job['pid']))
    return tuple(result)


def smoke_slot(rt,capacity):
    for seed in SEEDS:
        if not slot_ready(rt,seed):continue
        gpu=RESOURCE_GPU_BY_SEED[seed];apps,free=capacity(gpu,reserved_predecessor_pids(rt,seed))
        if apps<3 and free>=10000:return seed,gpu
    return None
