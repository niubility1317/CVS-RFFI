"""Only add rejected-U nonphysical RDrop to the fixed StepNoCap objective."""
from copy import deepcopy
from .hard_pl_step_no_cap import configuration as base_configuration,validate as base_validate,RUN as NC_RUN,METHODS as NC_METHODS
from .hard_pl_step_guard import RUN as SG_RUN,METHODS as SG_METHODS
from .hard_pl import SEEDS,RESOURCE_GPU_BY_SEED
from .rejected_rdrop import RECIPE

RUN='20261024-phase1-PAMF-HardPL-AllURDrop-m3-r01'
METHODS=('HPL_E20_AS_ALLRD',)


def configuration(template,project,commit):
    c=base_configuration(template,project,commit);seed=c['model_seed']
    c.update(run_id=RUN,row_id=METHODS[0]+'-s'+str(seed),algorithm_id='pamf_hard_pl_all_u_rdrop_v1',
        display_method='PAMF 全 U RDrop',hard_pl_all_u_rdrop=deepcopy(RECIPE),
        output_root=project+'/runs/'+RUN+'/'+METHODS[0]+'-s'+str(seed)+'/source')
    return validate(c)


def validate(c):
    seed=c.get('model_seed')
    if (c.get('run_id')!=RUN or c.get('row_id')!=METHODS[0]+'-s'+str(seed)
            or c.get('algorithm_id')!='pamf_hard_pl_all_u_rdrop_v1' or c.get('hard_pl_all_u_rdrop')!=RECIPE):
        raise ValueError('Only fixed rejected-U RDrop on the complete StepNoCap recipe allowed')
    proxy=deepcopy(c);proxy.pop('hard_pl_all_u_rdrop')
    proxy.update(run_id=NC_RUN,row_id=NC_METHODS[0]+'-s'+str(seed),algorithm_id='pamf_hard_pl_step_no_raw_cap_v1')
    base_validate(proxy);return c


def slot_ready(rt,seed):
    from experiments.cric_dura_comparison.io import read
    done=('PREDICTIONS_COMPLETE','SCORING','SCORED')
    old=read(rt['step_guard_GPU_owner_state'])['jobs'].get(SG_METHODS[0]+'-s'+str(seed),{})
    peer=read(rt['step_no_cap_GPU_owner_state'])['jobs'].get(NC_METHODS[0]+'-s'+str(seed),{})
    return old.get('status') in done and peer.get('status') in ('RUNNING',)+done


def reserved_predecessor_pids(rt,seed):
    from experiments.cric_dura_comparison.io import read
    result=[]
    for path,method in ((rt['step_guard_GPU_owner_state'],SG_METHODS[0]),(rt['step_no_cap_GPU_owner_state'],NC_METHODS[0])):
        job=read(path)['jobs'].get(method+'-s'+str(seed),{})
        if job.get('status')=='RUNNING':result.append(int(job['pid']))
    return tuple(result)


def smoke_slot(rt,capacity):
    for seed in SEEDS:
        if slot_ready(rt,seed):
            gpu=RESOURCE_GPU_BY_SEED[seed];apps,free=capacity(gpu,reserved_predecessor_pids(rt,seed))
            if apps<3 and free>=10000:return seed,gpu
    return None
