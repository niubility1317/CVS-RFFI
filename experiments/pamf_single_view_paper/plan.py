"""Fixed paper reproduction matrix; display names do not rename old artifacts."""
from copy import deepcopy

RUN='20261027-phase1-pamf-single-view-paper-m120-r01'
RELEASE=RUN+'-release'
SEEDS=(2026092701,2026092702,2026092703)
PROTOCOLS={
 'main_split':('F0','Main split','标准划分'),
 'receiver_split_1':('F1_rx_repartition_r01','Receiver split 1','接收机划分一'),
 'receiver_split_2':('F2_rx_repartition_r01','Receiver split 2','接收机划分二'),
 'ten_class':('ManyRx','10-class setting','十类设置'),
 'labeled_1pct':('label01','1% labeled','1% 标注'),
 'labeled_5pct':('label05','5% labeled','5% 标注'),
 **{f'source_rx_{m}':(f'm{m}-natural',f'{m} source receiver'+('s' if m!=1 else ''),f'{m} 个源接收机') for m in (1,2,3,4,6)}}
ARMS={
 'full':('Full PAMF','完整单视图 PAMF'),
 'no_time':('w/o Time coordinate','去显式时间坐标'),
 'no_signature':('w/o Temporal signature','去路径描述支路'),
 'no_pa':('w/o PA branch','去功放相关支路'),
 'no_reference':('w/o Reference response','去参考响应支路'),
 'moment_pooling':('Moment pooling','矩统计替代'),
 'shuffled_time':('Shuffled temporal order','固定打乱局部顺序'),
 'random_reference_phase':('Phase-randomized reference','参考模板相位打乱')}
TASKS=('cross_date','cross_rx','cross_both')
VIEWS=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
READOUTS=('raw','DURA-L','DURA-LU')

def make_plan():
    result=[]
    def add(setting,family,arm,regime,aug=True):
        for seed in SEEDS:
            rid=f'{setting}-{family}-{arm}-{regime}-'+('channel_aug' if aug else 'no_channel_aug')+f'-s{seed}'
            reuse=(family=='main' and setting=='main_split' and aug) or (family=='branch' and regime=='supervised' and arm in ('full','no_signature'))
            key=(f'main_split/full/{regime}/{seed}' if family=='main' else f'branch/{arm}/{regime}/{seed}') if reuse else None
            result.append(dict(row_id=rid,data_setting=setting,legacy_protocol_id=PROTOCOLS[setting][0],
                split_name=PROTOCOLS[setting][1],family=family,arm=arm,arm_name=ARMS[arm][0],
                learning_regime=regime,learning_name='Supervised' if regime=='supervised' else 'Semi-supervised (PL)',
                objective_name=('Full labeled objective' if family=='main' else 'CE only')+(' + ordinary PL' if regime=='pl' else ''),
                channel_augmentation=aug,model_seed=seed,signature_views=0 if arm=='no_signature' else 1,
                reuse_key=key,epochs=50,updates=11100,trajectory=[20,10,20],
                labeled_only_epochs=50 if regime=='supervised' else 13,joint_LU_epochs=0 if regime=='supervised' else 37,
                L_batch=128,U_batch=0 if regime=='supervised' else 256,
                views=list(VIEWS if aug else ('clean',)),readouts=list(READOUTS),tasks=list(TASKS)))
    # Finish and test each fixed row independently; all settings/seeds are kept.
    for setting in PROTOCOLS:
        for regime in ('supervised','pl'):add(setting,'main','full',regime)
    for regime in ('supervised','pl'):add('main_split','main','full',regime,False)
    for arm in ARMS:
        for regime in ('supervised','pl'):add('main_split','branch',arm,regime)
    assert len(result)==120 and len({r['row_id'] for r in result})==120
    assert sum(bool(r['reuse_key']) for r in result)==12
    return dict(run_id=RUN,rows=result,new_F=108,reused_F=12,logical_F=120,new_G=0,new_IQ_cache=0,
        main_rows=72,branch_rows=48,branch_structures=8,model_seeds=list(SEEDS),protocol_names=PROTOCOLS,
        scientific_selection='fixed full paper matrix; no target-driven gate, ranking, tuning or rerun',
        own_GPU_workers_per_card=1,total_GPU_processes_per_card=2,allowed_gpus=list(range(8)),CPU_scorers=1,
        target_truth='independent CPU scorer only after all row predictions freeze')

def configuration(context,row,project,commit):
    h=context['handles'][row['legacy_protocol_id']]
    c=deepcopy(context['templates'][str(row['model_seed'])]['base']['F21'])
    for k in list(c):
        if k.startswith('hard_pl') or k in ('shared_prefix','parent_source','step_offset','epoch_offset','segment_epochs','F_parent_run','replaces_source_row','supervised_architecture','core_mechanisms'):
            c.pop(k)
    use=row['learning_regime']=='pl';ce=row['family']=='branch'
    c.update(run_id=RUN,row_id=row['row_id'],paper_row=deepcopy(row),model_seed=row['model_seed'],
        protocol_id=row['legacy_protocol_id'],display_protocol=row['split_name'],paper_method='PAMF',
        algorithm_id='pamf_single_view_paper_v1',method_id='pamf_t1_e50',
        source_cache=h['source_cache'],source_contract=h['source_contract'],
        source_rxs=context['protocols'][row['legacy_protocol_id']]['source_rxs'],source_days=[1,2,3],num_classes=len(h['classes']),
        output_root=project+'/runs/'+RUN+'/F/'+row['row_id']+'/source',code_commit=commit,
        ablation='F15' if ce else ('F17' if not row['channel_augmentation'] else 'F21' if use else 'F20'),
        regime='matched_sat_plain_pl' if use else 'matched_sat_supervised',U_use=use,U_start_epoch=14,
        unlabeled_batch=256,labeled_batch=128,own_EMA=False,native_U_gate=False,native_unlabeled_loop=False,
        satellite_concat=ce,satellite_CE_weight=.68 if ce else 0.,channel_augmentation=row['channel_augmentation'],
        satellite_consistency_weight=0.,plain_PL_threshold=.95,plain_PL_weight=1.,
        augmentation_seed=2026102301,augmentation_cache=context['runtime']['augmentation_cache'],
        default_head_joint=True,posthoc_head_fit=False,epochs=50,optimizer_updates=11100,trajectory=[20,10,20],
        initialization='scratch_same_contract',G_source_roles='L_s+U_s_TX_hidden' if use else 'L_s')
    return validate_config(c)

def validate_config(c):
    row=c.get('paper_row',{})
    expected=next((r for r in make_plan()['rows'] if r['row_id']==c.get('row_id')),None)
    if row!=expected or c.get('run_id')!=RUN or c.get('model_seed')!=row['model_seed']:raise ValueError('Unregistered paper row')
    for k in ('truth','target_truth','tests','views_root','dataset_path','checkpoint','teacher_checkpoint','shared_prefix'):
        if c.get(k):raise ValueError('Source-only paper training required')
    ce=row['family']=='branch';use=row['learning_regime']=='pl'
    if any(c.get(k) for k in ('own_EMA','native_U_gate','native_unlabeled_loop','posthoc_head_fit','core_mechanisms','supervised_architecture')):raise ValueError('Ordinary PL and original joint head required')
    if c.get('U_use') is not use or c.get('regime')!=('matched_sat_plain_pl' if use else 'matched_sat_supervised'):raise ValueError('Wrong supervised/U permissions')
    if (c.get('ablation')!=('F15' if ce else ('F17' if not row['channel_augmentation'] else 'F21' if use else 'F20'))
        or c.get('satellite_concat') is not ce or c.get('satellite_CE_weight')!=(.68 if ce else 0.)
        or c.get('channel_augmentation') is not row['channel_augmentation']):raise ValueError('Objective or augmentation differs')
    if (c.get('epochs')!=50 or c.get('optimizer_updates')!=11100 or c.get('trajectory')!=[20,10,20]
        or c.get('U_start_epoch')!=14 or c.get('unlabeled_batch')!=256 or c.get('plain_PL_threshold')!=.95 or c.get('plain_PL_weight')!=1.
        or c.get('num_classes')!=(10 if row['data_setting']=='ten_class' else 6) or c.get('source_days')!=[1,2,3]):raise ValueError('Fixed budget/data/PL contract differs')
    if 'segment_epochs' in c:
        offset=c.get('epoch_offset');length={0:20,20:10,30:20}.get(offset)
        if c['segment_epochs']!=length or c.get('step_offset')!=offset*222:raise ValueError('Own E20+10+20 segments required')
    return c
