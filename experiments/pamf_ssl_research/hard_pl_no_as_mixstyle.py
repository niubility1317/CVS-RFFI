"""Factorial source-based style extension of the fixed no-U-channel objective."""
from copy import deepcopy
from .hard_pl_no_uas import RUN as NO_UAS_RUN,configuration as no_as_configuration,validate as no_as_validate
from .hard_pl_mixstyle import STYLE,RUN as MS_RUN
from .hard_pl_ls import RUN as LS_RUN
from .hard_pl import PARENT_RUN,SEEDS,RESOURCE_GPU_BY_SEED

RUN='20261024-phase1-PAMF-HardPL-NoAsMixStyle-m3-r01'
METHOD='HPL_NAMS2'


def configuration(template,project,commit):
    c=no_as_configuration(template,project,commit);seed=c['model_seed']
    c.update(run_id=RUN,row_id=METHOD+'-s'+str(seed),algorithm_id='pamf_hard_pl_no_as_mixstyle_v1',
        display_method='PAMF no-U-channel supervised hard PL plus accepted cross-RX style',
        hard_pl_mixstyle=deepcopy(STYLE),hard_pl_shared_style_plan=True,
        output_root=project+'/runs/'+RUN+'/'+METHOD+'-s'+str(seed)+'/source')
    return validate(c)


def validate(c):
    seed=c.get('model_seed')
    if (c.get('run_id')!=RUN or c.get('row_id')!=METHOD+'-s'+str(seed)
        or c.get('algorithm_id')!='pamf_hard_pl_no_as_mixstyle_v1' or c.get('hard_pl_mixstyle')!=STYLE
        or c.get('hard_pl_shared_style_plan') is not True):
        raise ValueError('Fixed no-U-channel,shared accepted-only source style profile required')
    proxy=deepcopy(c);proxy.pop('hard_pl_mixstyle');proxy.pop('hard_pl_shared_style_plan')
    proxy.update(run_id=NO_UAS_RUN,row_id='HPL_NO_UAS-s'+str(seed),algorithm_id='pamf_hard_pl_no_uas_v1')
    no_as_validate(proxy)
    return c


def slot_ready(rt,seed):
    from experiments.cric_dura_comparison.io import read
    done=('PREDICTIONS_COMPLETE','SCORING','SCORED')
    prior=read(rt['gpu_predecessor_state'])['jobs'];ms=read(rt['MS_GPU_owner_state'])['jobs']
    return (prior.get('HPL_LS2-s'+str(seed),{}).get('status') in done
        and ms.get('HPL_MS2-s'+str(seed),{}).get('status') in ('RUNNING',)+done)


def reserved_predecessor_pids(rt,seed):
    from experiments.cric_dura_comparison.io import read
    job=read(rt['MS_GPU_owner_state'])['jobs'].get('HPL_MS2-s'+str(seed),{})
    return (int(job['pid']),) if job.get('status')=='RUNNING' else ()
