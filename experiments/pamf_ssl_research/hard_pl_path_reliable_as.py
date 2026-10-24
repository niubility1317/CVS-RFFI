"""Source-motivated factorial combination of fixed Path CE and ReliableAs inputs."""
from copy import deepcopy
from .hard_pl_path import configuration as path_configuration,validate as path_validate,RUN as PATH_RUN,METHODS as PATH_METHODS
from .hard_pl_reliable_as import RELIABLE_AS,validate as ras_validate,RUN as RAS_RUN,METHODS as RAS_METHODS
from .hard_pl import SEEDS,RESOURCE_GPU_BY_SEED

RUN='20261024-phase1-PAMF-HardPL-PathReliableAs-m3-r01'
METHODS=('HPL_E20_PATH_RAS6G',)


def configuration(template,project,commit):
    c=path_configuration(template,project,commit);seed=c['model_seed'];method=METHODS[0]
    c.update(run_id=RUN,row_id=method+'-s'+str(seed),algorithm_id='pamf_hard_pl_path_reliable_as_v1',
        display_method='PAMF 主干互补与可靠信道视图',hard_pl_reliable_as=deepcopy(RELIABLE_AS),hard_pl_no_uas=False,
        output_root=project+'/runs/'+RUN+'/'+method+'-s'+str(seed)+'/source')
    return validate(c)


def validate(c):
    seed=c.get('model_seed')
    if (c.get('run_id')!=RUN or c.get('row_id')!=METHODS[0]+'-s'+str(seed)
        or c.get('algorithm_id')!='pamf_hard_pl_path_reliable_as_v1'):
        raise ValueError('Fixed source-only Path/ReliableAs factorial method required')
    proxy=deepcopy(c);proxy.pop('hard_pl_path',None)
    proxy.update(run_id=RAS_RUN,row_id=RAS_METHODS[0]+'-s'+str(seed),algorithm_id='pamf_hard_pl_reliable_as_v1')
    ras_validate(proxy)
    proxy=deepcopy(c);proxy.pop('hard_pl_reliable_as',None)
    proxy.update(run_id=PATH_RUN,row_id=PATH_METHODS[0]+'-s'+str(seed),algorithm_id='pamf_hard_pl_path_complement_v1',hard_pl_no_uas=True)
    path_validate(proxy);return c


def slot_ready(rt,seed):
    from experiments.cric_dura_comparison.io import read
    done=('PREDICTIONS_COMPLETE','SCORING','SCORED')
    old=read(rt['path_GPU_owner_state'])['jobs'].get(PATH_METHODS[0]+'-s'+str(seed),{})
    peer=read(rt['reliable_as_GPU_owner_state'])['jobs'].get(RAS_METHODS[0]+'-s'+str(seed),{})
    return old.get('status') in done and peer.get('status') in ('RUNNING',)+done


def reserved_predecessor_pids(rt,seed):
    from experiments.cric_dura_comparison.io import read
    result=[]
    for path,method in ((rt['path_GPU_owner_state'],PATH_METHODS[0]),(rt['reliable_as_GPU_owner_state'],RAS_METHODS[0])):
        job=read(path)['jobs'].get(method+'-s'+str(seed),{})
        if job.get('status')=='RUNNING':result.append(int(job['pid']))
    return tuple(result)


def smoke_slot(rt,capacity):
    for seed in SEEDS:
        if not slot_ready(rt,seed):continue
        gpu=RESOURCE_GPU_BY_SEED[seed];apps,free=capacity(gpu,reserved_predecessor_pids(rt,seed))
        if apps<3 and free>=10000:return seed,gpu
    return None
