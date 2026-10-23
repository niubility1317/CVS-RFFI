"""Stronger ordinary pseudo supervision with the complete supervised gradient as anchor."""
from copy import deepcopy
from .hard_pl_early import configuration as early_configuration,validate as early_validate,RUN as EARLY_RUN,METHODS as EARLY_METHODS
from .hard_pl import SEEDS,RESOURCE_GPU_BY_SEED

RUN='20261024-phase1-PAMF-HardPL-GradientGuard-m6-r01'
METHODS=('HPL_E20_LS6G','HPL_E20_NAMS6G')
GUARD=dict(weight=6.,projection='one_sided_per_group',groups=['t2','cls_head','other'],
    U_to_L_norm_cap=1.,tiny_L_squared_norm=1e-12,anchor='complete_current_L_objective',
    optimizer='unchanged_native_Adam',extra_teacher_or_student_forward=False)


def configuration(template,method,project,commit):
    if method not in METHODS:raise ValueError('Fixed stronger guarded ordinary-PL method required')
    c=early_configuration(template,EARLY_METHODS[METHODS.index(method)],project,commit)
    seed=c['model_seed'];c['hard_pl']['weight']=6.
    c.update(run_id=RUN,row_id=method+'-s'+str(seed),algorithm_id='pamf_'+method.lower()+'_v1',
        display_method='PAMF '+method,hard_pl_gradient_guard=deepcopy(GUARD),
        output_root=project+'/runs/'+RUN+'/'+method+'-s'+str(seed)+'/source')
    return validate(c)


def validate(c):
    method=c.get('row_id','').rsplit('-s',1)[0];seed=c.get('model_seed')
    if (method not in METHODS or c.get('run_id')!=RUN or c.get('row_id')!=method+'-s'+str(seed)
        or c.get('algorithm_id')!='pamf_'+method.lower()+'_v1' or c.get('hard_pl',{}).get('weight')!=6.
        or c.get('hard_pl_gradient_guard')!=GUARD):
        raise ValueError('Fixed weight6 supervised-anchored ordinary-PL guard required')
    proxy=deepcopy(c);proxy.pop('hard_pl_gradient_guard');proxy['hard_pl']['weight']=2.
    previous=EARLY_METHODS[METHODS.index(method)]
    proxy.update(run_id=EARLY_RUN,row_id=previous+'-s'+str(seed),algorithm_id='pamf_'+previous.lower()+'_v1')
    early_validate(proxy)
    return c


def slot_ready(rt,seed,method,jobs):
    from experiments.cric_dura_comparison.io import read
    old=read(rt['early_GPU_owner_state'])['jobs'];done=('PREDICTIONS_COMPLETE','SCORING','SCORED')
    ls=old.get(EARLY_METHODS[0]+'-s'+str(seed),{});nams=old.get(EARLY_METHODS[1]+'-s'+str(seed),{})
    if method==METHODS[0]:return ls.get('status') in done and nams.get('status') in ('RUNNING',)+done
    return (method==METHODS[1] and ls.get('status') in done and nams.get('status') in done
        and jobs.get(METHODS[0]+'-s'+str(seed),{}).get('status') in ('RUNNING',)+done)


def reserved_predecessor_pids(rt,seed):
    from experiments.cric_dura_comparison.io import read
    jobs=read(rt['early_GPU_owner_state'])['jobs']
    return tuple(int(jobs[key]['pid']) for method in EARLY_METHODS
        if (key:=method+'-s'+str(seed)) in jobs and jobs[key].get('status')=='RUNNING')
