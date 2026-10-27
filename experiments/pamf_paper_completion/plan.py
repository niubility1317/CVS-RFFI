"""The fixed, deduplicated paper matrix. No target-driven selection."""
from copy import deepcopy
from experiments.pamf_single_view_paper.plan import PROTOCOLS,TASKS,VIEWS,READOUTS,SEEDS

RUN='20261027-phase1-pamf-paper-completion-m119-r01'
RELEASE=RUN+'-release'
CE_VARIANT='pamf_tri_sigtime_ce'

def make_plan():
    rows=[]
    def add(setting,method,structure='full',permuted=False,family='factorial',reuse=None,views=3,aug=True):
        for seed in SEEDS:
            pid=PROTOCOLS[setting][0]
            key=(reuse.format(pid=pid,seed=seed) if reuse else None)
            rows.append(dict(row_id=f'{setting}-{family}-{method}-{structure}-'+('permuted' if permuted else 'ordered')+f'-s{seed}',
                data_setting=setting,protocol_id=pid,split_name=PROTOCOLS[setting][1],family=family,
                method=method,structure=structure,permuted=permuted,signature_views=views,channel_augmentation=aug,
                learning_regime='supervised' if method.startswith('sup') else 'pl' if method.startswith('pl') else 'ssl',
                model_seed=seed,epochs=50,updates=11100,views=list(VIEWS),readouts=list(READOUTS),
                reuse_key=key,new_training=not bool(key)))
    for setting in PROTOCOLS:
        add(setting,'sup',reuse='{pid}-supervised-s{seed}')
        add(setting,'pl',reuse='{pid}-ordinary_PL-s{seed}')
        add(setting,'sup_noaug',aug=False)
        add(setting,'pl_noaug',aug=False)
        add(setting,'ssl',permuted=True,reuse='current/3/{seed}' if setting=='main_split' else None)
    add('main_split','ssl',permuted=True,family='structure',views=1,reuse='current/1/{seed}')
    add('main_split','ssl',family='structure',reuse='F0-REPL-s{seed}')
    add('main_split','ssl',structure='no_signature',family='structure',views=0,reuse='F0-REPL_NO_SIGNATURE-s{seed}')
    add('main_split','ssl',structure='moment',family='structure',reuse='F0-REPL_COV_X3-s{seed}')
    for arm in ('no_time','no_pa','no_reference','reference_phase'):
        add('main_split','ssl',structure=arm,permuted=True,family='structure')
    add('main_split','sup',permuted=True,family='U_permission')
    for seed in range(2026092701,2026092709):
        rows.append(dict(row_id=f'main_split-architecture-tri_sigtime-CE-E200-s{seed}',data_setting='main_split',protocol_id='F0',
            split_name='Main split',family='architecture',method='ce',learning_regime='supervised',structure='full',
            signature_views=3,permuted=False,channel_augmentation=False,model_seed=seed,epochs=200,updates=10000,
            views=['clean'],readouts=['raw'],reuse_key=None,new_training=True))
    assert len(rows)==200 and len({r['row_id'] for r in rows})==200
    assert sum(r['new_training'] for r in rows)==119
    assert not any(r['learning_regime']=='pl' and r['permuted'] for r in rows)
    return dict(run_id=RUN,rows=rows,logical_F=200,new_F=119,reused_F=81,new_G=0,new_IQ_cache=0,
                new_E50=111,new_E200=8,reused_full_readouts=75,partial_raw_reuses=6,
                target_feedback=False,selection='fixed final; all declared rows and seeds retained')

def configuration(ctx,row,project,commit):
    if row not in make_plan()['rows']:raise ValueError('Unregistered scientific row')
    if row['family']=='architecture':
        c=deepcopy(ctx['ce_template'])
        c.update(run_id=RUN,row_id=row['row_id'],paper_row=deepcopy(row),variant=CE_VARIANT,
                 model_seed=row['model_seed'],output_root=project+'/runs/'+RUN+'/F/'+row['row_id']+'/source',code_commit=commit)
        return validate_config(c)
    pid=row['protocol_id'];seed=str(row['model_seed']);h=ctx['handles'][pid]
    c=deepcopy(ctx['templates'][seed]['base']['F21'])
    for k in list(c):
        if k.startswith('hard_pl') or k in ('shared_prefix','parent_source','step_offset','epoch_offset','segment_epochs','F_parent_run','replaces_source_row','core_mechanisms','supervised_architecture','view_comparison','repl_row'):
            c.pop(k)
    use=row['learning_regime']!='supervised';ssl=row['learning_regime']=='ssl'
    c.update(run_id=RUN,row_id=row['row_id'],paper_row=deepcopy(row),model_seed=row['model_seed'],protocol_id=pid,
        source_cache=ctx['templates'][seed]['base']['F21']['source_cache'] if pid=='F0' else h['source_cache'],
        source_contract=ctx['templates'][seed]['base']['F21']['source_contract'] if pid=='F0' else h['source_contract'],
        source_rxs=ctx['protocols'][pid]['source_rxs'],source_days=[1,2,3],num_classes=len(h['classes']),
        output_root=project+'/runs/'+RUN+'/F/'+row['row_id']+'/source',code_commit=commit,
        paper_method='PAMF / Tri-SIG-TIME SSL' if ssl else 'PAMF-'+('Sup' if not use else 'PL'),
        algorithm_id='pamf_paper_completion_v1',method_id='pamf_t1_e50',
        ablation='F17' if not row['channel_augmentation'] else 'F21' if use else 'F20',
        regime='matched_sat_plain_pl' if use else 'matched_sat_supervised',U_use=use,U_start_epoch=14,
        unlabeled_batch=256,labeled_batch=128,own_EMA=False,native_U_gate=False,native_unlabeled_loop=False,
        satellite_concat=False,satellite_CE_weight=0.,satellite_consistency_weight=0.,plain_PL_threshold=.95,plain_PL_weight=1.,
        G_source_roles='L_s+U_s_TX_hidden' if use else 'L_s',
        default_head_joint=True,posthoc_head_fit=False,epochs=50,optimizer_updates=11100,trajectory=[20,10,20])
    if ssl:
        ref=ctx['templates'][seed]['repl']
        for k,v in ref.items():
            if k.startswith('hard_pl'):c[k]=deepcopy(v)
        c['repl_row']=dict(U=True,L_A=True,L_B=True,blocks=dict(style=True,guard=True,joint=True,extrapolation=True))
    return validate_config(c)

def validate_config(c):
    row=c.get('paper_row');expected=next((r for r in make_plan()['rows'] if r['row_id']==c.get('row_id')),None)
    if row!=expected or expected is None or c.get('run_id')!=RUN or c.get('model_seed')!=row['model_seed']:raise ValueError('Unregistered row identity')
    for k in ('truth','target_truth','tests','views_root','target_inputs','initial_checkpoint','checkpoint','teacher_checkpoint','shared_prefix','fork_parent_source'):
        if c.get(k):raise ValueError('Source-only scratch contract rejects '+k)
    if row['family']=='architecture':
        wanted=dict(method='standard_ce_baselines',variant=CE_VARIANT,epochs=200,batch_size=128,lr=.0002,lr_min=1e-6,
                    weight_decay=.0001,drop_last=False,augmentation=False,domain_backbone=False,extra_losses=[],selection='fixed_last_epoch',split_seed=392005)
        if any(c.get(k)!=v for k,v in wanted.items()):raise ValueError('Matched E200 pure CE differs')
        return c
    use=row['learning_regime']!='supervised';ssl=row['learning_regime']=='ssl'
    wanted=dict(method_id='pamf_t1_e50',U_use=use,regime='matched_sat_plain_pl' if use else 'matched_sat_supervised',
        ablation='F17' if not row['channel_augmentation'] else 'F21' if use else 'F20',epochs=50,optimizer_updates=11100,
        trajectory=[20,10,20],U_start_epoch=14,unlabeled_batch=256,labeled_batch=128,satellite_concat=False,
        satellite_CE_weight=0.,plain_PL_threshold=.95,plain_PL_weight=1.,default_head_joint=True,posthoc_head_fit=False,
        own_EMA=False,native_U_gate=False,native_unlabeled_loop=False,source_days=[1,2,3],num_classes=10 if row['data_setting']=='ten_class' else 6)
    if any(c.get(k)!=v for k,v in wanted.items()):raise ValueError('Fixed E50 method/budget/U permissions differ')
    if bool(c.get('hard_pl'))!=ssl or c.get('core_mechanisms') or c.get('supervised_architecture'):raise ValueError('Mixed training methods')
    if ssl and (c['hard_pl']['threshold']!=.95 or c['hard_pl']['weight']!=6. or not c['hard_pl_no_uas'] or
                c['repl_row']['blocks']!=dict(style=True,guard=True,joint=True,extrapolation=True)):raise ValueError('Complete Tri-SIG-TIME SSL required')
    if 'segment_epochs' in c:
        offset=c.get('epoch_offset');n={0:20,20:10,30:20}.get(offset)
        if c['segment_epochs']!=n or c.get('step_offset')!=offset*222:raise ValueError('Own fixed E20+10+20 continuation required')
    return c
