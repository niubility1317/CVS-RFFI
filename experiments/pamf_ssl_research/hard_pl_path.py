"""Dose-matched full,signature-masked and core-only pseudo CE with one existing head."""
from copy import deepcopy
from .hard_pl_joint import configuration as base_configuration,validate as base_validate,RUN as JOINT_RUN,METHODS as JOINT_METHODS
from .hard_pl_complement import RUN as COMP_RUN,METHODS as COMP_METHODS
from .hard_pl import SEEDS,RESOURCE_GPU_BY_SEED

RUN='20261024-phase1-PAMF-HardPL-PathComplement-m3-r01'
METHODS=('HPL_E20_JOINT_PATH6G',)
PATH=dict(features=['actual_joint160','joint_minus_same_pass_sequence','joint_minus_same_pass_sequence_and_scaled_response'],
    ordinary_CE_coefficient=.25,no_signature_CE_coefficient=.125,core_CE_coefficient=.125,
    ordinary_CE_coefficient_before=.5,CE_normalizer=1.25,label_smoothing=.01,
    student_views='first_two_ordinary_U_passes',extra_existing_head_calls=4,extra_backbone_forwards=0,new_head=False)


def configuration(template,project,commit):
    c=base_configuration(template,project,commit);seed=c['model_seed'];method=METHODS[0]
    c.update(run_id=RUN,row_id=method+'-s'+str(seed),algorithm_id='pamf_hard_pl_path_complement_v1',
        display_method='PAMF 主干互补伪标签监督',hard_pl_path=deepcopy(PATH),
        output_root=project+'/runs/'+RUN+'/'+method+'-s'+str(seed)+'/source')
    return validate(c)


def validate(c):
    seed=c.get('model_seed')
    if (c.get('run_id')!=RUN or c.get('row_id')!=METHODS[0]+'-s'+str(seed)
        or c.get('algorithm_id')!='pamf_hard_pl_path_complement_v1' or c.get('hard_pl_path')!=PATH):
        raise ValueError('Fixed full/no-signature/core-only same-head CE dose required')
    proxy=deepcopy(c);proxy.pop('hard_pl_path')
    proxy.update(run_id=JOINT_RUN,row_id=JOINT_METHODS[0]+'-s'+str(seed),algorithm_id='pamf_hard_pl_joint160_v1')
    base_validate(proxy);return c


def slot_ready(rt,seed):
    from experiments.cric_dura_comparison.io import read
    done=('PREDICTIONS_COMPLETE','SCORING','SCORED')
    old=read(rt['joint_GPU_owner_state'])['jobs'].get(JOINT_METHODS[0]+'-s'+str(seed),{})
    peer=read(rt['complement_GPU_owner_state'])['jobs'].get(COMP_METHODS[0]+'-s'+str(seed),{})
    return old.get('status') in done and peer.get('status') in ('RUNNING',)+done


def reserved_predecessor_pids(rt,seed):
    from experiments.cric_dura_comparison.io import read
    result=[]
    for path,method in ((rt['joint_GPU_owner_state'],JOINT_METHODS[0]),(rt['complement_GPU_owner_state'],COMP_METHODS[0])):
        job=read(path)['jobs'].get(method+'-s'+str(seed),{})
        if job.get('status')=='RUNNING':result.append(int(job['pid']))
    return tuple(result)


def smoke_slot(rt,capacity):
    for seed in SEEDS:
        if not slot_ready(rt,seed):continue
        gpu=RESOURCE_GPU_BY_SEED[seed];apps,free=capacity(gpu,reserved_predecessor_pids(rt,seed))
        if apps<3 and free>=10000:return seed,gpu
    return None
