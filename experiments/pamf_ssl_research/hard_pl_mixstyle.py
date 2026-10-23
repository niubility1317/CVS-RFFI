"""Accepted hard-PL identities protect cross-RX shallow/middle style partners."""
from copy import deepcopy
from .hard_pl_ls import configuration as ls_configuration,validate as ls_validate,RUN as LS_RUN
from .hard_pl import PARENT_RUN,SEEDS,RESOURCE_GPU_BY_SEED
from .hard_pl_no_uas import RUN as NO_UAS_RUN

RUN='20261024-phase1-PAMF-HardPL-MixStyle-m3-r01'
METHOD='HPL_MS2'
STYLE=dict(levels=['time_down','t2.pool'],alpha=.1,probability_per_level=.5,eps=1e-6,
    accepted_only=True,same_hard_pseudo=True,different_source_RX=True,detached_moments=True)


def configuration(template,project,commit):
    c=ls_configuration(template,project,commit);seed=c['model_seed']
    c.update(run_id=RUN,row_id=METHOD+'-s'+str(seed),algorithm_id='pamf_hard_pl_mixstyle_v1',
        display_method='PAMF hard PL LS with accepted same-identity cross-RX MixStyle',
        hard_pl_mixstyle=deepcopy(STYLE),output_root=project+'/runs/'+RUN+'/'+METHOD+'-s'+str(seed)+'/source')
    return validate(c)


def validate(c):
    seed=c.get('model_seed')
    if (c.get('run_id')!=RUN or c.get('row_id')!=METHOD+'-s'+str(seed)
        or c.get('algorithm_id')!='pamf_hard_pl_mixstyle_v1' or c.get('hard_pl_mixstyle')!=STYLE):
        raise ValueError('Fixed accepted-only same-pseudo cross-RX style recipe required')
    proxy=deepcopy(c);proxy.pop('hard_pl_mixstyle')
    proxy.update(run_id=LS_RUN,row_id='HPL_LS2-s'+str(seed),algorithm_id='pamf_hard_pl_ls_v1')
    ls_validate(proxy)
    return c


def slot_ready(rt,seed):
    from experiments.cric_dura_comparison.io import read
    done=('PREDICTIONS_COMPLETE','SCORING','SCORED')
    hard=read(rt['hard_PL_GPU_owner_state'])['jobs']
    previous=read(rt['gpu_predecessor_state'])['jobs']
    ls=read(rt['LS_GPU_owner_state'])['jobs']
    return (all(hard.get(m+'-s'+str(seed),{}).get('status') in done for m in ('HPL2','HPLT1U'))
        and previous.get('HPL_NO_UAS-s'+str(seed),{}).get('status') in done
        and ls.get('HPL_LS2-s'+str(seed),{}).get('status') in ('RUNNING',)+done)


def reserved_predecessor_pids(rt,seed):
    from experiments.cric_dura_comparison.io import read
    job=read(rt['LS_GPU_owner_state'])['jobs'].get('HPL_LS2-s'+str(seed),{})
    return (int(job['pid']),) if job.get('status')=='RUNNING' else ()
