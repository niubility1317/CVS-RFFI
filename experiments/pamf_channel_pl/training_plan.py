"""Fixed all-contract candidate, preserving the matched ordinary-PL schedule."""
from copy import deepcopy
from .plan import PROTOCOLS,SEEDS,GPU_BY_SEED
from .policy import RECIPE,validate_recipe

TRAIN_RUN='20261025-phase1-PAMF-ChannelPL-m33-r01'


def rows():
    return [dict(row_id=p+'-ChannelPL-s'+str(s),protocol=p,model_seed=s,gpu=GPU_BY_SEED[s],
        method='ChannelPL',new_F=True,epochs=50,new_epochs=30,updates=11100,new_updates=6660,
        trajectory=[20,10,20],readouts=['raw','DURA-LU']) for p in PROTOCOLS for s in SEEDS]


def make_config(case,project,commit):
    c=deepcopy(case['source_config']);pid=case['protocol'];seed=case['model_seed'];rid=pid+'-ChannelPL-s'+str(seed)
    for key in list(c):
        if key.startswith('hard_pl') or key in ('parent_source','step_offset','epoch_offset','segment_epochs',
            'F_parent_run','initialization','replaces_source_row','status','core_mechanisms'):
            c.pop(key)
    row=dict(row_id=rid,protocol_id=pid,model_seed=seed,method='ChannelPL',group='main',
        L_A=True,L_B=True,U=True,structure='full',blocks=dict(style=False,joint=False,guard=False,extrapolation=False))
    c.update(run_id=TRAIN_RUN,row_id=rid,protocol_id=pid,model_seed=seed,method_id='pamf_t1_e50',
        algorithm_id='pamf_channel_pl_v1',paper_method='PAMF-ChannelPL',repl_row=row,channel_pl=dict(RECIPE),
        output_root=project+'/runs/'+TRAIN_RUN+'/F/'+rid+'/source',code_commit=commit,
        shared_prefix=case['parent_source'],expected_source_contract=case['expected_source_contract'],
        regime='matched_sat_plain_pl',ablation='F21',U_use=True,U_start_epoch=14,unlabeled_batch=256,labeled_batch=128,
        own_EMA=False,native_U_gate=False,native_unlabeled_loop=False,plain_PL_threshold=.95,plain_PL_weight=1.,
        satellite_concat=False,satellite_CE_weight=0.,satellite_consistency_weight=0.,
        default_head_joint=True,posthoc_head_fit=False,epochs=50,optimizer_updates=11100,trajectory=[20,10,20],
        source_E20_reference_row=case['row_id'],source_E20_reference_config=deepcopy(case['source_config']))
    validate_config(c)
    return c


def validate_config(c):
    validate_recipe(c['channel_pl']);p=c['protocol_id'];s=c['model_seed']
    if (p not in PROTOCOLS or s not in SEEDS or c['run_id']!=TRAIN_RUN
        or c['row_id']!=p+'-ChannelPL-s'+str(s) or c['method_id']!='pamf_t1_e50'
        or c['algorithm_id']!='pamf_channel_pl_v1' or c['ablation']!='F21'
        or c['num_classes']!=(10 if p=='ManyRx' else 6)):
        raise ValueError('Registered matching ChannelPL row/class space required')
    if (any(c.get(k) for k in ('truth','target_truth','target_inputs','views_root','tests','dataset_path','checkpoint','teacher_checkpoint'))
            or any(k.startswith('hard_pl') for k in c)):
        raise ValueError('Source-only ordinary-PL scaffold required; no REPL objective or query inputs')
    if (c['regime']!='matched_sat_plain_pl' or c['own_EMA'] or c['native_U_gate'] or c['native_unlabeled_loop']
        or c['satellite_concat'] or c['satellite_CE_weight'] or c['satellite_consistency_weight']
        or c['posthoc_head_fit'] or not c['default_head_joint'] or not c['U_use']
        or c['plain_PL_threshold']!=.95 or c['plain_PL_weight']!=1. or c['U_start_epoch']!=14
        or c['labeled_batch']!=128 or c['unlabeled_batch']!=256 or c['source_days']!=[1,2,3]
        or c['epochs']!=50 or c['optimizer_updates']!=11100 or c['trajectory']!=[20,10,20]
        or not c['shared_prefix'].endswith('/segment20')):
        raise ValueError('Exact default head,source schedule,budget and dose required')
    r=c['repl_row']
    if (not r['L_A'] or not r['L_B'] or not r['U'] or r['structure']!='full' or r['group']!='main'
        or any(r['blocks'].values())):
        raise ValueError('Original supervised mechanisms and no REPL U blocks required')
    return c
