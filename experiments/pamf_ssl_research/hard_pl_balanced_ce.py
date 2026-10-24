"""Source-fixed bounded CE contribution balancing on the full native-As recipe."""
from copy import deepcopy
from .hard_pl_native_as import configuration as base_configuration,validate as base_validate,RUN as NAS_RUN,METHODS as NAS_METHODS
from .hard_pl_receiver_extrapolation import METHODS as RXE_METHODS
from .hard_pl import SEEDS,RESOURCE_GPU_BY_SEED

RUN='20261024-phase1-PAMF-HardPL-BalancedCE-m3-r01'
METHODS=('HPL_E20_AS_BCE6G',)
BALANCED_CE=dict(counts='current_raw_accepted_pseudo_class_and_actual_source_RX',
    inverse_frequency='sqrt(A/(C*n_class)*A/(R*n_RX))',normalize='mean_accepted_one',
    mixing='min(0.5,0.5/(max(normalized_ratio)-1))',minimum=.5,maximum=1.5,
    scope='CE_only_same_weights_on_all_six_existing_student_logits',raw_gate=.95,
    ordinary_pair_coefficient=.5,mixed_pair_coefficient=.5,raw_pair_coefficient=.25,CE_normalizer=1.25,
    label_smoothing=.01,extra_CE_kernels_per_step=6,extra_teacher=0,extra_backbone=0,extra_head=0,
    persistent_statistics=False,new_head=False)


def configuration(template,project,commit):
    c=base_configuration(template,project,commit);seed=c['model_seed']
    c.update(run_id=RUN,row_id=METHODS[0]+'-s'+str(seed),algorithm_id='pamf_hard_pl_balanced_ce_v1',
        display_method='PAMF 高权重伪标签类别与接收机贡献均衡',hard_pl_balanced_ce=deepcopy(BALANCED_CE),
        output_root=project+'/runs/'+RUN+'/'+METHODS[0]+'-s'+str(seed)+'/source')
    return validate(c)


def validate(c):
    seed=c.get('model_seed')
    if (c.get('run_id')!=RUN or c.get('row_id')!=METHODS[0]+'-s'+str(seed)
        or c.get('algorithm_id')!='pamf_hard_pl_balanced_ce_v1' or c.get('hard_pl_balanced_ce')!=BALANCED_CE):
        raise ValueError('Fixed bounded current accepted CE contribution recipe required')
    proxy=deepcopy(c);proxy.pop('hard_pl_balanced_ce')
    proxy.update(run_id=NAS_RUN,row_id=NAS_METHODS[0]+'-s'+str(seed),algorithm_id='pamf_hard_pl_native_as_v1')
    base_validate(proxy);return c


def slot_ready(rt,seed):
    from experiments.cric_dura_comparison.io import read
    done=('PREDICTIONS_COMPLETE','SCORING','SCORED')
    old=read(rt['receiver_extrapolation_GPU_owner_state'])['jobs'].get(RXE_METHODS[0]+'-s'+str(seed),{})
    peer=read(rt['native_as_GPU_owner_state'])['jobs'].get(NAS_METHODS[0]+'-s'+str(seed),{})
    return old.get('status') in done and peer.get('status') in ('RUNNING',)+done


def reserved_predecessor_pids(rt,seed):
    from experiments.cric_dura_comparison.io import read
    result=[]
    for path,method in ((rt['receiver_extrapolation_GPU_owner_state'],RXE_METHODS[0]),(rt['native_as_GPU_owner_state'],NAS_METHODS[0])):
        job=read(path)['jobs'].get(method+'-s'+str(seed),{})
        if job.get('status')=='RUNNING':result.append(int(job['pid']))
    return tuple(result)


def smoke_slot(rt,capacity):
    for seed in SEEDS:
        if not slot_ready(rt,seed):continue
        gpu=RESOURCE_GPU_BY_SEED[seed];apps,free=capacity(gpu,reserved_predecessor_pids(rt,seed))
        if apps<3 and free>=10000:return seed,gpu
    return None
