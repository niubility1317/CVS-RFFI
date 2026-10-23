"""Keep shallow/middle U contrast and add a direct deployed-joint160 objective."""
from copy import deepcopy
from .hard_pl_guard import configuration as guard_configuration,validate as guard_validate,RUN as GUARD_RUN,METHODS as GUARD_METHODS
from .hard_pl import SEEDS,RESOURCE_GPU_BY_SEED

RUN='20261024-phase1-PAMF-HardPL-Joint160-m3-r01'
METHODS=('HPL_E20_NAMS_JOINT6G',)
JOINT=dict(feature='actual_CompositeReference_features_joint160',weight=.05,CE_normalizer=1.25,
    student_views='first_two_ordinary_U_passes',accepted_only=True,positive='same_pseudo_different_source_RX',
    contrast='same_native_CROSS_RX_temperature0.2',extra_forward=False,new_head=False)


def configuration(template,project,commit):
    c=guard_configuration(template,GUARD_METHODS[1],project,commit);seed=c['model_seed'];method=METHODS[0]
    c.update(run_id=RUN,row_id=method+'-s'+str(seed),algorithm_id='pamf_hard_pl_joint160_v1',
        display_method='PAMF 伪标签浅中层及融合特征跨RX',hard_pl_joint_contrast=deepcopy(JOINT),
        output_root=project+'/runs/'+RUN+'/'+method+'-s'+str(seed)+'/source')
    return validate(c)


def validate(c):
    seed=c.get('model_seed')
    if (c.get('run_id')!=RUN or c.get('row_id')!=METHODS[0]+'-s'+str(seed)
        or c.get('algorithm_id')!='pamf_hard_pl_joint160_v1' or c.get('hard_pl_joint_contrast')!=JOINT):
        raise ValueError('Fixed default-head joint160 plus existing shallow/middle U objective required')
    proxy=deepcopy(c);proxy.pop('hard_pl_joint_contrast')
    proxy.update(run_id=GUARD_RUN,row_id=GUARD_METHODS[1]+'-s'+str(seed),algorithm_id='pamf_'+GUARD_METHODS[1].lower()+'_v1')
    guard_validate(proxy)
    return c


def slot_ready(rt,seed):
    from experiments.cric_dura_comparison.io import read
    jobs=read(rt['guard_GPU_owner_state'])['jobs'];done=('PREDICTIONS_COMPLETE','SCORING','SCORED')
    return (jobs.get(GUARD_METHODS[0]+'-s'+str(seed),{}).get('status') in done
        and jobs.get(GUARD_METHODS[1]+'-s'+str(seed),{}).get('status') in ('RUNNING',)+done)


def reserved_predecessor_pids(rt,seed):
    from experiments.cric_dura_comparison.io import read
    jobs=read(rt['guard_GPU_owner_state'])['jobs']
    return tuple(int(jobs[key]['pid']) for method in GUARD_METHODS
        if (key:=method+'-s'+str(seed)) in jobs and jobs[key].get('status')=='RUNNING')
