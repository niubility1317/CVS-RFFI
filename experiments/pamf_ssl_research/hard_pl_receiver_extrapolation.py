"""Source-calibrated complete-feature extrapolation under unchanged Joint6G training."""
from copy import deepcopy
from .hard_pl_joint import configuration as base_configuration,validate as base_validate,RUN as JOINT_RUN,METHODS as JOINT_METHODS
from .hard_pl_reliable_as import METHODS as RAS_METHODS
from .hard_pl_path_reliable_as import METHODS as COMBO_METHODS
from .hard_pl import SEEDS,RESOURCE_GPU_BY_SEED

RUN='20261024-phase1-PAMF-HardPL-ReceiverExtrap-m3-r01'
METHODS=('HPL_E20_JOINT_RXE6G',)
RECEIVER_EXTRAPOLATION=dict(feature='actual_CompositeReference_features_joint160',alpha=.25,
    ordinary_CE_coefficient_before=.5,ordinary_CE_coefficient=.25,extrapolated_CE_coefficient=.25,
    CE_normalizer=1.25,label_smoothing=.01,student_views='first_two_ordinary_U_passes',
    partner='reuse_same_shared_six_U_style_partner_and_valid',invalid_partner='original_CE_no_delta_no_sample_drop',
    extra_existing_head_calls=2,extra_backbone_forwards=0,new_head=False)


def configuration(template,project,commit):
    c=base_configuration(template,project,commit);seed=c['model_seed'];method=METHODS[0]
    c.update(run_id=RUN,row_id=method+'-s'+str(seed),algorithm_id='pamf_hard_pl_receiver_extrapolation_v1',
        display_method='PAMF 完整特征接收机外推',hard_pl_receiver_extrapolation=deepcopy(RECEIVER_EXTRAPOLATION),
        output_root=project+'/runs/'+RUN+'/'+method+'-s'+str(seed)+'/source')
    return validate(c)


def validate(c):
    seed=c.get('model_seed')
    if (c.get('run_id')!=RUN or c.get('row_id')!=METHODS[0]+'-s'+str(seed)
        or c.get('algorithm_id')!='pamf_hard_pl_receiver_extrapolation_v1'
        or c.get('hard_pl_receiver_extrapolation')!=RECEIVER_EXTRAPOLATION):
        raise ValueError('Fixed source-calibrated .25 full-feature extrapolation and conserved CE dose required')
    proxy=deepcopy(c);proxy.pop('hard_pl_receiver_extrapolation')
    proxy.update(run_id=JOINT_RUN,row_id=JOINT_METHODS[0]+'-s'+str(seed),algorithm_id='pamf_hard_pl_joint160_v1')
    base_validate(proxy);return c


def slot_ready(rt,seed):
    from experiments.cric_dura_comparison.io import read
    done=('PREDICTIONS_COMPLETE','SCORING','SCORED')
    old=read(rt['reliable_as_GPU_owner_state'])['jobs'].get(RAS_METHODS[0]+'-s'+str(seed),{})
    peer=read(rt['path_reliable_as_GPU_owner_state'])['jobs'].get(COMBO_METHODS[0]+'-s'+str(seed),{})
    return old.get('status') in done and peer.get('status') in ('RUNNING',)+done


def reserved_predecessor_pids(rt,seed):
    from experiments.cric_dura_comparison.io import read
    result=[]
    for path,method in ((rt['reliable_as_GPU_owner_state'],RAS_METHODS[0]),(rt['path_reliable_as_GPU_owner_state'],COMBO_METHODS[0])):
        job=read(path)['jobs'].get(method+'-s'+str(seed),{})
        if job.get('status')=='RUNNING':result.append(int(job['pid']))
    return tuple(result)


def smoke_slot(rt,capacity):
    for seed in SEEDS:
        if not slot_ready(rt,seed):continue
        gpu=RESOURCE_GPU_BY_SEED[seed];apps,free=capacity(gpu,reserved_predecessor_pids(rt,seed))
        if apps<3 and free>=10000:return seed,gpu
    return None
