"""Single canonical matrix: matching supervised/PL/REPL and focused ablations."""
from copy import deepcopy

RUN='20261024-phase1-PAMF-REPL-completion-m201-r01'
SEEDS=(2026092701,2026092702,2026092703)
PROTOCOLS=('F0','F1_rx_repartition_r01','F2_rx_repartition_r01','ManyRx','label01','label05','m1-natural','m2-natural','m3-natural','m4-natural','m6-natural')
BLOCKS=('style','guard','joint','extrapolation')
STRUCTURES=('full','SIG_TIME','NO_TIME','SHUFFLE_TIME','NO_SIGNATURE','NO_PA','NO_REFERENCE','WRONG_REFERENCE','COV_X3')
FINE={'L_shallow_only':'F01','L_middle_only':'F02','L_joint_only':'F03','L_single_forward':'F05',
      'L_without_As':'F17','L_As_KL_on':'F18','L_without_raw_CE':'F19'}

def make_plan():
    rows=[]
    def add(pid,method,seed,group,**extra):
        rows.append(dict(row_id=f'{pid}-{method}-s{seed}',protocol_id=pid,method=method,
            model_seed=seed,group=group,G_roles='L' if method=='supervised' else 'LU',
            reuse=pid=='F0' and method in ('supervised','ordinary_PL','REPL','without_extrapolation'),**extra))
    for pid in PROTOCOLS:
        for method in ('supervised','ordinary_PL','REPL'):
            for seed in SEEDS:add(pid,method,seed,'main',structure='full',blocks={k:True for k in BLOCKS})
    for block in BLOCKS:
        for seed in SEEDS:add('F0','without_'+block,seed,'mechanism',structure='full',blocks={k:k!=block for k in BLOCKS})
    for bits in ('000','001','010','011','100','101'):
        for seed in SEEDS:add('F0','ABC'+bits,seed,'core',structure='full',blocks={k:True for k in BLOCKS},L_A=bits[0]=='1',L_B=bits[1]=='1',U=bits[2]=='1')
    for method in FINE:
        for seed in SEEDS:add('F0',method,seed,'fine_L',structure='full',blocks={k:True for k in BLOCKS})
    for structure in STRUCTURES:
        if structure!='full':
            for seed in SEEDS:add('F0','REPL_'+structure,seed,'structure',structure=structure,blocks={k:True for k in BLOCKS})
        for seed in SEEDS:add('F0','CE_'+structure,seed,'strict_structure',structure=structure,blocks={k:False for k in BLOCKS},U=False,L_A=False,L_B=False)
    for row in rows:
        row.setdefault('U',row['method']!='supervised');row.setdefault('L_A',True);row.setdefault('L_B',True)
        row['G_roles']='LU' if row['U'] else 'L'
        row['readouts']=['raw','DURA-L','DURA-LU']
    p=dict(run_id=RUN,rows=rows,model_seeds=list(SEEDS),protocols=list(PROTOCOLS),structure_arms=list(STRUCTURES),logical_rows=201,
        reused_F=12,new_F=189,new_G=0,new_IQ_cache=0,predictions=201,scores=201,
        main_rows=99,mechanism_rows=12,core_additional_rows=18,fine_L_rows=21,structure_additional_rows=24,strict_structure_rows=27,
        new_unique_model_epochs=8670,new_unique_updates=8670*222,shared_E20_prefixes=84,
        G_component_readouts=21,G_crossed_additional_readouts=6,G_component_new_G=0,total_prediction_jobs=228,total_score_jobs=228,
        dedicated_timing_case_rounds=dict(REPL_structure=450,strict_structure=450,main_controls=100,G_combinations=175),
        allowed_gpus=list(range(8)),max_total_GPU_workers=24,max_per_GPU=3,
        excluded=['external_method_retraining','new_G_training','fixed_L_budget','m5','independent_chrono_training','other_fine_ablations'],
        full_recipe_fixed_before_new_target_access=True,posthoc_historical_method_selection_disclosed=True)
    validate(p);return p

def validate(p):
    rows=p['rows']
    if len(rows)!=201 or len({r['row_id'] for r in rows})!=201:raise ValueError('Fixed 201-row matrix required')
    if sum(r['reuse'] for r in rows)!=12:raise ValueError('Only 12 exact-contract existing models may be reused')
    if set(p['allowed_gpus'])!=set(range(8)):raise ValueError('All eight GPUs are authorized')
    for r in rows:
        if r['protocol_id'] not in PROTOCOLS or r['model_seed'] not in SEEDS:raise ValueError('Unregistered contract/seed')
        if r['G_roles']!=('LU' if r['U'] else 'L'):raise ValueError('Wrong restoration source permissions')
        if r['reuse'] and (r['protocol_id']!='F0' or r['method'] not in ('supervised','ordinary_PL','REPL','without_extrapolation')):raise ValueError('Cross-contract reuse forbidden')
    return p

def configuration(context,row,project,commit):
    seed=str(row['model_seed']);pid=row['protocol_id'];h=context['handles'][pid]
    base=context['templates'][seed]['base']['F21'];c=deepcopy(base)
    # These internal wrapper fields retain the original segmented native trainer;
    # actual scientific method identity is separate from that compatibility ID.
    c.update(run_id=RUN,row_id=row['row_id'],protocol_id=pid,paper_method='PAMF-REPL' if row['method']=='REPL' else row['method'],
        algorithm_id='pamf_repl_completion_v1',repl_row=deepcopy(row),method_id='pamf_t1_e50',
        source_rxs=context['protocols'][pid]['source_rxs'],source_days=[1,2,3],num_classes=len(h['classes']),
        source_cache=base['source_cache'] if pid=='F0' else h['source_cache'],
        source_contract=base['source_contract'] if pid=='F0' else h['source_contract'],
        output_root=project+'/runs/'+RUN+'/F/'+row['row_id']+'/source',code_commit=commit,
        ablation='F16' if not row['L_A'] else FINE.get(row['method'],'F21' if row['U'] else 'F20'),
        regime='matched_sat_plain_pl' if row['U'] else 'matched_sat_supervised',
        U_use=row['U'],own_EMA=False,native_U_gate=False,native_unlabeled_loop=False,
        U_start_epoch=14,unlabeled_batch=256,plain_PL_threshold=.95,plain_PL_weight=1.,
        satellite_concat=False,satellite_CE_weight=0.,satellite_consistency_weight=0.,
        default_head_joint=True,posthoc_head_fit=False,epochs=50,optimizer_updates=11100,trajectory=[20,10,20])
    for k in ('parent_source','step_offset','epoch_offset','segment_epochs','F_parent_run','initialization','replaces_source_row'):
        c.pop(k,None)
    selected=context['templates'][seed]['repl']
    for k in ('hard_pl','hard_pl_no_uas','hard_pl_label_smoothing','hard_pl_mixstyle','hard_pl_shared_style_plan',
              'hard_pl_gradient_guard','hard_pl_joint_contrast','hard_pl_receiver_extrapolation'):
        if k in selected:c[k]=deepcopy(selected[k])
    if not row['blocks']['style']:c['hard_pl_mixstyle']['probability_per_level']=0.
    if not row['blocks']['joint']:c['hard_pl_joint_contrast']['weight']=0.
    if row['group']=='strict_structure':
        c.update(ablation='F15',satellite_concat=True,satellite_CE_weight=.68,channel_augmentation=True,
                 augmentation_seed=2026102301,augmentation_cache=context['runtime']['augmentation_cache'],
                 source_cache=h['source_cache'],source_contract=h['source_contract'])
    validate_config(c,context)
    return c

def validate_config(c,context=None):
    row=c['repl_row']
    expected=next((r for r in make_plan()['rows'] if r['row_id']==c['row_id']),None)
    if row!=expected or c['run_id']!=RUN or c['model_seed']!=row['model_seed']:raise ValueError('Source config not in registered matrix')
    if any(c.get(k) for k in ('truth','target_truth','views_root','tests','dataset_path','teacher_checkpoint','checkpoint')):raise ValueError('Training must be source-only')
    if c['source_days']!=[1,2,3] or c['num_classes']!=(10 if row['protocol_id']=='ManyRx' else 6):raise ValueError('Data contract/classes differ')
    if c['own_EMA'] or c['satellite_concat']!=(row['group']=='strict_structure') or c['posthoc_head_fit'] or c['epochs']!=50 or c['optimizer_updates']!=11100:raise ValueError('Fixed method and budget required')
    if c['U_use']!=row['U'] or c['U_start_epoch']!=14:raise ValueError('Matched source exposures differ')
    if c['hard_pl']['weight']!=6. or c['hard_pl']['threshold']!=.95 or c['hard_pl_receiver_extrapolation']['alpha']!=.25:raise ValueError('Selected REPL parameters changed')
    if context:
        contract=context['protocols'][row['protocol_id']]
        if c['source_rxs']!=contract['source_rxs']:raise ValueError('Source receiver contract mismatch')
    return c
