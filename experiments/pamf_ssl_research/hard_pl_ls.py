"""Isolate supervised label smoothing on high-purity ordinary hard PL."""
from copy import deepcopy
from .hard_pl import PARENT_RUN,SEEDS,RESOURCE_GPU_BY_SEED,configuration as base_configuration,validate as base_validate,RUN as HARD_RUN
from .hard_pl_no_uas import RUN as NO_UAS_RUN

RUN='20261024-phase1-PAMF-HardPL-LS-m3-r01'
METHOD='HPL_LS2'


def configuration(template,project,commit):
    c=base_configuration(template,'HPL2',project,commit);seed=c['model_seed']
    c.update(run_id=RUN,row_id=METHOD+'-s'+str(seed),algorithm_id='pamf_hard_pl_ls_v1',
        display_method='PAMF hard PL weight2 with supervised label smoothing',hard_pl_label_smoothing=.01,
        output_root=project+'/runs/'+RUN+'/'+METHOD+'-s'+str(seed)+'/source')
    return validate(c)


def validate(c):
    seed=c.get('model_seed')
    if (c.get('run_id')!=RUN or c.get('row_id')!=METHOD+'-s'+str(seed)
        or c.get('algorithm_id')!='pamf_hard_pl_ls_v1' or c.get('hard_pl_label_smoothing')!=.01
        or c.get('hard_pl',{}).get('method')!='HPL2'):
        raise ValueError('Fixed ordinary hard-PL label-smoothing profile required')
    proxy=deepcopy(c);proxy.pop('hard_pl_label_smoothing')
    proxy.update(run_id=HARD_RUN,row_id='HPL2-s'+str(seed),algorithm_id='pamf_hpl2_v1')
    base_validate(proxy)
    return c


def slot_ready(rt,seed):
    from experiments.cric_dura_comparison.io import read
    done=('PREDICTIONS_COMPLETE','SCORING','SCORED')
    no_as=read(rt['gpu_predecessor_state'])['jobs'];hard=read(rt['hard_PL_GPU_owner_state'])['jobs']
    return (no_as.get('HPL_NO_UAS-s'+str(seed),{}).get('status') in done
        and hard.get('HPLT1U-s'+str(seed),{}).get('status') in ('RUNNING',)+done)


def reserved_predecessor_pids(rt,seed):
    from experiments.cric_dura_comparison.io import read
    job=read(rt['hard_PL_GPU_owner_state'])['jobs'].get('HPLT1U-s'+str(seed),{})
    return (int(job['pid']),) if job.get('status')=='RUNNING' else ()
