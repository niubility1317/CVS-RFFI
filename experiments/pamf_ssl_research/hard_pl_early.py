"""Earlier high-purity U supervision:fixed original F21 E20 plus native10+20."""
from copy import deepcopy
from .hard_pl import PARENT_RUN,SEEDS,RESOURCE_GPU_BY_SEED
from .hard_pl_ls import configuration as ls_configuration,validate as ls_validate,RUN as LS_RUN
from .hard_pl_no_as_mixstyle import configuration as nams_configuration,validate as nams_validate,RUN as NAMS_RUN
from .hard_pl_mixstyle import RUN as MS_RUN

RUN='20261024-phase1-PAMF-HardPL-E20plus30-m6-r01'
METHODS=('HPL_E20_LS2','HPL_E20_NAMS2')


def configuration(template,method,project,commit):
    if method not in METHODS:raise ValueError('Fixed earlier-exposure matched method required')
    c=(ls_configuration if method==METHODS[0] else nams_configuration)(template,project,commit)
    seed=c['model_seed'];c['hard_pl']['start_epoch']=21
    c.update(run_id=RUN,row_id=method+'-s'+str(seed),algorithm_id='pamf_'+method.lower()+'_v1',
        display_method='PAMF '+method,hard_pl_parent_epoch=20,hard_pl_continuation_epochs=30,
        output_root=project+'/runs/'+RUN+'/'+method+'-s'+str(seed)+'/source')
    return validate(c)


def validate(c):
    method=c.get('row_id','').rsplit('-s',1)[0];seed=c.get('model_seed')
    if (method not in METHODS or c.get('run_id')!=RUN or c.get('row_id')!=method+'-s'+str(seed)
        or c.get('algorithm_id')!='pamf_'+method.lower()+'_v1' or c.get('hard_pl_parent_epoch')!=20
        or c.get('hard_pl_continuation_epochs')!=30 or c.get('hard_pl',{}).get('start_epoch')!=21):
        raise ValueError('Original F21 E20 plus fixed native10+20 earlier-U schedule required')
    proxy=deepcopy(c);proxy.pop('hard_pl_parent_epoch');proxy.pop('hard_pl_continuation_epochs')
    proxy['hard_pl']['start_epoch']=31
    if method==METHODS[0]:
        proxy.update(run_id=LS_RUN,row_id='HPL_LS2-s'+str(seed),algorithm_id='pamf_hard_pl_ls_v1');ls_validate(proxy)
    else:
        proxy.update(run_id=NAMS_RUN,row_id='HPL_NAMS2-s'+str(seed),algorithm_id='pamf_hard_pl_no_as_mixstyle_v1');nams_validate(proxy)
    return c


def slot_ready(rt,seed,method,jobs):
    from experiments.cric_dura_comparison.io import read
    done=('PREDICTIONS_COMPLETE','SCORING','SCORED')
    ms=read(rt['gpu_predecessor_state'])['jobs'].get('HPL_MS2-s'+str(seed),{})
    nams=read(rt['NAMS_GPU_owner_state'])['jobs'].get('HPL_NAMS2-s'+str(seed),{})
    if method==METHODS[0]:return ms.get('status') in done and nams.get('status') in ('RUNNING',)+done
    return (method==METHODS[1] and ms.get('status') in done and nams.get('status') in done
        and jobs.get(METHODS[0]+'-s'+str(seed),{}).get('status') in ('RUNNING',)+done)


def reserved_predecessor_pids(rt,seed):
    from experiments.cric_dura_comparison.io import read
    job=read(rt['NAMS_GPU_owner_state'])['jobs'].get('HPL_NAMS2-s'+str(seed),{})
    return (int(job['pid']),) if job.get('status')=='RUNNING' else ()
