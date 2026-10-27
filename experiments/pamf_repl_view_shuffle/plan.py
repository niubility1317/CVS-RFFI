from copy import deepcopy

RUN='20261027-phase1-pamf-repl-shuffled-views-m16-r01'
RELEASE=RUN+'-release'
SEEDS=tuple(range(2026092701,2026092709))
TASKS=('cross_rx','cross_both')
VIEWS=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
ORDER_SEED=2026102207

def make_plan():
    rows=[]
    # Complete the entire three-view group before any single-view training.
    for views in (3,1):
        for seed in SEEDS:
            rows.append(dict(row_id=f'main_split-repl-shuffled_time-{views}view-s{seed}',model_seed=seed,view_count=views,
                display_name=f'Main split · {views} signature view'+('s' if views==3 else '')+' · REPL · Shuffled temporal order'))
    return dict(run_id=RUN,rows=rows,new_F=16,reused_F=0,new_G=0,epochs=50,optimizer_updates=11100,
        model_seeds=list(SEEDS),tasks=list(TASKS),views=list(VIEWS),readouts=['raw'],selection='fixed_final_E50',
        matching='Only number of independently learned temporal signature descriptors differs',priority='Complete all three-view training/prediction/scoring before single-view training',
        excluded=['ordinary_PL_paper_queue','DURA_training','DURA_readouts','target_feedback_selection'])

def configuration(ctx,row,project,commit):
    if row not in make_plan()['rows']:raise ValueError('Unregistered view/seed row')
    seed=str(row['model_seed']);old=ctx['historical_shuffle'][seed if seed in ctx['historical_shuffle'] else '2026092701']['config']
    c=deepcopy(old)
    for k in ('shared_prefix','parent_source','step_offset','epoch_offset','segment_epochs','F_parent_run','initialization','replaces_source_row'):
        c.pop(k,None)
    rr=deepcopy(old['repl_row']);rr.update(row_id=row['row_id'],model_seed=row['model_seed'],reuse=False)
    c.update(run_id=RUN,row_id=row['row_id'],model_seed=row['model_seed'],paper_method='PAMF-REPL shuffled temporal order',
        algorithm_id='pamf_repl_shuffled_views_v1',repl_row=rr,view_comparison=deepcopy(row),
        output_root=project+'/runs/'+RUN+'/F/'+row['row_id']+'/source',code_commit=commit)
    validate_config(c,ctx)
    return c

def validate_config(c,ctx=None):
    row=c.get('view_comparison');rr=c.get('repl_row',{})
    if row not in make_plan()['rows'] or c['run_id']!=RUN or c['row_id']!=row['row_id'] or c['model_seed']!=row['model_seed']:
        raise ValueError('Fixed view comparison identity required')
    if any(c.get(k) for k in ('truth','target_truth','tests','views_root','dataset_path','checkpoint','teacher_checkpoint','shared_prefix','fork_parent_source')):
        raise ValueError('Scratch source-only training; no foreign checkpoint or target')
    if (rr.get('structure')!='SHUFFLE_TIME' or rr.get('L_A') is not True or rr.get('L_B') is not True or rr.get('U') is not True
        or rr.get('blocks')!=dict(style=True,guard=True,joint=True,extrapolation=True)):
        raise ValueError('Complete original REPL plus fixed time shuffle required')
    if (c['protocol_id']!='F0' or c['source_rxs']!=[1,3,4,6,8] or c['source_days']!=[1,2,3] or c['num_classes']!=6
        or c['epochs']!=50 or c['optimizer_updates']!=11100 or c['trajectory']!=[20,10,20]
        or not c['U_use'] or c['own_EMA'] or c['native_U_gate'] or c['native_unlabeled_loop']
        or c['satellite_concat'] or c['posthoc_head_fit'] or not c['default_head_joint'] or c['ablation']!='F21'
        or c['U_start_epoch']!=14 or c['unlabeled_batch']!=256 or c['plain_PL_threshold']!=.95 or c['plain_PL_weight']!=1.):
        raise ValueError('Historical matched data, budget and ordinary-PL prefix required')
    if c['hard_pl']['weight']!=6. or c['hard_pl']['threshold']!=.95 or not c['hard_pl_no_uas']:
        raise ValueError('Original REPL U weight, gate and raw-As branch required')
    if ctx:
        reference=ctx['historical_shuffle']['2026092701']['config']
        for key in reference:
            if key.startswith('hard_pl') and c.get(key)!=reference[key]:raise ValueError('Frozen REPL mechanism changed: '+key)
        if c['source_contract']!=reference['source_contract'] or c['source_cache']!=reference['source_cache']:
            raise ValueError('Historical training contract/physical roles differ')
    return c
