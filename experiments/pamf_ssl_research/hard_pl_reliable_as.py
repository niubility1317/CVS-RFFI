"""Confidence/agreement selects a source As view;otherwise retain its exact raw IQ."""
from copy import deepcopy
from .hard_pl_joint import configuration as base_configuration,validate as base_validate,RUN as JOINT_RUN,METHODS as JOINT_METHODS
from .hard_pl_complement import METHODS as COMP_METHODS
from .hard_pl_path import METHODS as PATH_METHODS
from .hard_pl import SEEDS,RESOURCE_GPU_BY_SEED

RUN='20261024-phase1-PAMF-HardPL-ReliableAs-m3-r01'
METHODS=('HPL_E20_JOINT_RAS6G',)
RELIABLE_AS=dict(raw_pseudo_gate=.95,augmented_view_gate=.95,require_same_raw_pseudo=True,
    require_raw_accepted=True,fallback='exact_raw_same_packet',teacher='current_student_eval_no_grad',
    physical_half='same_original_alternating_half',student_U_passes=6,extra_teacher_As_passes=1,
    new_pseudo_labels=False,CE_and_KL_coefficients_unchanged=True,CE_normalizer=1.25,new_head=False)


def configuration(template,project,commit):
    c=base_configuration(template,project,commit);seed=c['model_seed'];method=METHODS[0]
    c.update(run_id=RUN,row_id=method+'-s'+str(seed),algorithm_id='pamf_hard_pl_reliable_as_v1',
        display_method='PAMF 可靠信道视图与原始输入回退',hard_pl_reliable_as=deepcopy(RELIABLE_AS),hard_pl_no_uas=False,
        output_root=project+'/runs/'+RUN+'/'+method+'-s'+str(seed)+'/source')
    return validate(c)


def validate(c):
    seed=c.get('model_seed')
    if (c.get('run_id')!=RUN or c.get('row_id')!=METHODS[0]+'-s'+str(seed)
        or c.get('algorithm_id')!='pamf_hard_pl_reliable_as_v1' or c.get('hard_pl_reliable_as')!=RELIABLE_AS
        or c.get('hard_pl_no_uas') is not False):
        raise ValueError('Fixed same-pseudo source As eligibility/exactraw fallback and unchanged objective required')
    proxy=deepcopy(c);proxy.pop('hard_pl_reliable_as')
    proxy.update(run_id=JOINT_RUN,row_id=JOINT_METHODS[0]+'-s'+str(seed),algorithm_id='pamf_hard_pl_joint160_v1',hard_pl_no_uas=True)
    base_validate(proxy);return c


def slot_ready(rt,seed):
    from experiments.cric_dura_comparison.io import read
    done=('PREDICTIONS_COMPLETE','SCORING','SCORED')
    old=read(rt['complement_GPU_owner_state'])['jobs'].get(COMP_METHODS[0]+'-s'+str(seed),{})
    peer=read(rt['path_GPU_owner_state'])['jobs'].get(PATH_METHODS[0]+'-s'+str(seed),{})
    return old.get('status') in done and peer.get('status') in ('RUNNING',)+done


def reserved_predecessor_pids(rt,seed):
    from experiments.cric_dura_comparison.io import read
    result=[]
    for path,method in ((rt['complement_GPU_owner_state'],COMP_METHODS[0]),(rt['path_GPU_owner_state'],PATH_METHODS[0])):
        job=read(path)['jobs'].get(method+'-s'+str(seed),{})
        if job.get('status')=='RUNNING':result.append(int(job['pid']))
    return tuple(result)


def smoke_slot(rt,capacity):
    for seed in SEEDS:
        if not slot_ready(rt,seed):continue
        gpu=RESOURCE_GPU_BY_SEED[seed];apps,free=capacity(gpu,reserved_predecessor_pids(rt,seed))
        if apps<3 and free>=10000:return seed,gpu
    return None
