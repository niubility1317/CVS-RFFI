"""Single-factor U-channel ablation of the fixed supervised-hard-PL objective."""
from copy import deepcopy
from .hard_pl import PARENT_RUN,SEEDS,RESOURCE_GPU_BY_SEED,configuration as base_configuration,validate as base_validate,RUN as HARD_RUN
from .u_viewbalance import RUN as UVB_RUN

RUN='20261024-phase1-PAMF-HardPL-noUAs-m3-r01'
METHOD='HPL_NO_UAS'


def configuration(template,project,commit):
    c=base_configuration(template,'HPLT1U',project,commit);seed=c['model_seed']
    c.update(run_id=RUN,row_id=METHOD+'-s'+str(seed),algorithm_id='pamf_hard_pl_no_uas_v1',
        display_method='PAMF hard PL supervised U without U channel transform',hard_pl_no_uas=True,
        output_root=project+'/runs/'+RUN+'/'+METHOD+'-s'+str(seed)+'/source')
    return validate(c)


def validate(c):
    seed=c.get('model_seed')
    if (c.get('run_id')!=RUN or c.get('row_id')!=METHOD+'-s'+str(seed)
        or c.get('algorithm_id')!='pamf_hard_pl_no_uas_v1' or c.get('hard_pl_no_uas') is not True
        or c.get('hard_pl',{}).get('method')!='HPLT1U'):
        raise ValueError('Fixed no-U-As single-factor profile required')
    proxy=deepcopy(c);proxy.pop('hard_pl_no_uas')
    proxy.update(run_id=HARD_RUN,row_id='HPLT1U-s'+str(seed),algorithm_id='pamf_hplt1u_v1')
    base_validate(proxy)
    return c


def slot_ready(rt,seed):
    from experiments.cric_dura_comparison.io import read
    done=('PREDICTIONS_COMPLETE','SCORING','SCORED')
    hard=read(rt['gpu_predecessor_state'])['jobs'];uvb=read(rt['uvb_GPU_owner_state'])['jobs']
    return (hard.get('HPL2-s'+str(seed),{}).get('status') in done
        and hard.get('HPLT1U-s'+str(seed),{}).get('status') in ('RUNNING',)+done
        and uvb.get('UVB-s'+str(seed),{}).get('status') in done)


def reserved_predecessor_pids(rt,seed):
    """A dispatched HPLT1U reserves capacity before CUDA registration."""
    from experiments.cric_dura_comparison.io import read
    job=read(rt['gpu_predecessor_state'])['jobs'].get('HPLT1U-s'+str(seed),{})
    return (int(job['pid']),) if job.get('status')=='RUNNING' else ()
