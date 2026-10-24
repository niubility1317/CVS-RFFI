"""User-directed hard-PL weight and supervised-mechanism matched E30 forks."""
from copy import deepcopy
from .plan import SEEDS
from .view_anchor import RESOURCE_GPU_BY_SEED

RUN='20261024-phase1-PAMF-HardPL-T1U-m6-r01'
PARENT_RUN='20261022-phase1-T1-DURA-F-ablation-binding-m63-r01'
METHODS=('HPL2','HPLT1U')
RECIPE=dict(threshold=.95,weight=2.,teacher='current_student_eval_no_update',hard_targets=True,
    EMA=False,prototype=False,U_truth=False,start_epoch=31,default_head_joint=True,
    ordinary_noise_std=.015,T1_U_CE_normalizer=1.25,LS=.01,ordinary_raw_KL=1.,As_KL=0.,
    contrast_weight=.1,contrast_levels=['time_down','t2.pool'],contrast_only_accepted=True,
    physical_fraction=.5,contrast_same_pseudo_different_RX=True)


def configuration(template,method,project,commit):
    c=deepcopy(template);seed=c['model_seed']
    if method not in METHODS or c.get('run_id')!=PARENT_RUN or c.get('row_id')!='F21-s'+str(seed):
        raise ValueError('Original same-seed ordinary PL source required')
    parent=c['output_root']
    c.update(run_id=RUN,row_id=method+'-s'+str(seed),algorithm_id='pamf_'+method.lower()+'_v1',
        display_method='PAMF '+method,method_id='pamf_t1_e50',scaffold_id='pamf_t1_e50_supervised',
        hard_pl=dict(RECIPE,method=method),own_EMA=False,native_U_gate=False,native_unlabeled_loop=False,
        plain_PL_threshold=.95,plain_PL_weight=2.,U_use=True,U_start_epoch=14,unlabeled_batch=256,
        default_head_joint=True,posthoc_head_fit=False,fork_parent_source=parent,
        output_root=project+'/runs/'+RUN+'/'+method+'-s'+str(seed)+'/source',code_commit=commit)
    for key in ('parent_source','step_offset','epoch_offset','segment_epochs','F_parent_run'):
        c.pop(key,None)
    return validate(c)


def validate(c):
    method=c.get('hard_pl',{}).get('method');seed=c.get('model_seed')
    forbidden=('truth','target_truth','target_inputs','views_root','tests','dataset_path','checkpoint',
        'teacher_checkpoint','ssl','view_anchor','paired_u','u_cric','u_mixstyle','u_gradprotect',
        'u_cmr','u_dualanchor','u_viewbalance','core_mechanisms','supervised_architecture','hard_pl_no_uas','hard_pl_label_smoothing','hard_pl_mixstyle','hard_pl_shared_style_plan','hard_pl_parent_epoch','hard_pl_continuation_epochs','hard_pl_gradient_guard','hard_pl_joint_contrast','hard_pl_complement','hard_pl_path','hard_pl_reliable_as','hard_pl_receiver_extrapolation','hard_pl_native_as','hard_pl_balanced_ce')
    if (any(c.get(k) for k in forbidden) or method not in METHODS or seed not in SEEDS
        or c.get('hard_pl')!=dict(RECIPE,method=method) or c.get('run_id')!=RUN
        or c.get('row_id')!=method+'-s'+str(seed) or c.get('algorithm_id')!='pamf_'+method.lower()+'_v1'
        or c.get('method_id')!='pamf_t1_e50' or c.get('ablation')!='F21' or c.get('regime')!='native_cric'
        or c.get('satellite_concat') is not False or c.get('own_EMA') is not False
        or c.get('num_classes')!=6 or c.get('source_rxs')!=[1,3,4,6,8] or c.get('source_days')!=[1,2,3]
        or c.get('epochs')!=50 or c.get('optimizer_updates')!=11100 or c.get('trajectory')!=[20,10,20]
        or c.get('U_start_epoch')!=14 or c.get('posthoc_head_fit') is not False
        or not c.get('fork_parent_source','').endswith('/runs/'+PARENT_RUN+'/F21-s'+str(seed)+'/source')):
        raise ValueError('Fixed ordinary-PL parent,source handles,default head and E50 budget required')
    return c


def slot_ready(rt,seed,method,jobs):
    from experiments.cric_dura_comparison.io import read
    done=('PREDICTIONS_COMPLETE','SCORING','SCORED')
    udap=read(rt['udap_GPU_owner_state'])['jobs'].get('UDAP-s'+str(seed),{})
    uvb=read(rt['gpu_predecessor_state'])['jobs'].get('UVB-s'+str(seed),{})
    if method=='HPL2':
        return udap.get('status') in done and uvb.get('status') in ('RUNNING',)+done
    return (method=='HPLT1U' and uvb.get('status') in done and udap.get('status') in done
        and jobs.get('HPL2-s'+str(seed),{}).get('status') in ('RUNNING',)+done)
