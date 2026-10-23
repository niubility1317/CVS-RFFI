"""One result-independent matrix, consumed by registration and execution."""
from copy import deepcopy
from experiments.phase1_external_dg.plan import native_row, SEEDS, VIEWS
from experiments.cric_dura_comparison.plan import make_plan as previous_plan

RUN='20261023-phase1-practical-residual-full2x2-m600-r01'
FAMILIES=('riei','drift','ibrff','sdg')
CELLS={'A':(False,False),'B':(False,True),'C':(True,False),'D':(True,True)}
SCENES=('practical_low_suburban','practical_mid_urban','practical_low_urban')
SCENE_PROBABILITIES=(.40,.35,.25)
TASKS=('cross_date','cross_rx','cross_both')
NEW_RX={'F1_rx_repartition_r01':[0,1,2,5,7], 'F2_rx_repartition_r01':[0,3,9,10,11]}

def make_plan():
    old=previous_plan();protocols={}
    for pid in ('F0',*NEW_RX,'ManyRx','label01','label05',*[f'm{m}-natural' for m in (1,2,3,4,6)]):
        p=deepcopy(old['protocols']['F0' if pid in NEW_RX else pid])
        p.update(protocol_id=pid,fixed_labels=None,fixed_validation=None,diagnostic_day=0,
                 label_per_thousand=7 if pid=='label01' else 35 if pid=='label05' else 70)
        if pid in NEW_RX:
            p.update(source_rxs=NEW_RX[pid],target_rxs=[i for i in range(12) if i not in NEW_RX[pid]])
        protocols[pid]=p
    rows=[];reused=[]
    original=next(r for r in old['rows'] if r['protocol_id']=='F0' and r['regime']=='native_cric')
    supervised=next(r for r in old['rows'] if r['protocol_id']=='F0' and r['regime']=='matched_sat_supervised')
    for pid in protocols:
        for cell,(aug,pl) in CELLS.items():
            for family in FAMILIES:
                for seed in SEEDS:
                    c=native_row(pid,pid,family,'full',seed)
                    c.update(row_id=f'{pid}-{family}-{cell}-s{seed}',cell=cell,kind='external',
                        training_regime='pl' if pl else 'supervised',U_s_use=pl,
                        satellite_train_augmentation=aug,lambda_sat_cls=.68,pl_threshold=.95,pl_weight=1.,
                        augmentation_seed=2026102301,G_roles='LU' if pl else 'L',
                        augmentation_implementation='leo_practical_v3/residual/post_sync/noeq',
                        native_augmentation_retained=True,auxiliary_input='original_L_IQ',
                        ce_exposure_matched=True,precision='FP32/TF32off')
                    rows.append(c)
        for aug in (False,True):
            for seed in SEEDS:
                c=deepcopy(supervised)
                c.update(row_id=f'{pid}-PAMF-Sup-{"C" if aug else "A"}-s{seed}',protocol_id=pid,
                    model_seed=seed,kind='pamf',family='pamf',channel_augmentation=aug,
                    cell='C' if aug else 'A',G_roles='L',status='PLANNED_NOT_LAUNCHED',
                    initialization='scratch matching contract',F_parent_run=None)
                rows.append(c)
        for seed in SEEDS:
            c=deepcopy(original)
            c.update(row_id=f'{pid}-PAMF-T1-s{seed}',protocol_id=pid,model_seed=seed,
                kind='pamf',family='pamf',G_roles='LU',cell='native_T1',
                status='PLANNED_NOT_LAUNCHED' if pid in NEW_RX else 'FROZEN_RESULT_REUSE',
                initialization='scratch matching contract' if pid in NEW_RX else 'exact-contract historical T1',F_parent_run=None)
            (rows if pid in NEW_RX else reused).append(c)
    g=[dict(row_id=f'{pid}-G-{role}-s2026092701',protocol_id=pid,roles=role,
        model_seed=2026092701,stages=8,epochs_per_stage=20,updates_per_stage=4940,
        TX_labels_read=0,F_reads=0,target_reads=0) for pid in NEW_RX for role in ('L','LU')]
    p=dict(schema='practical_full_factorial_v1',run_id=RUN,protocols=protocols,rows=rows,
        reused_rows=reused,G_rows=g,views=list(VIEWS),tasks=list(TASKS),
        scene_probabilities=dict(zip(SCENES,SCENE_PROBABILITIES)),
        augmentation_schedule=[dict(start=0.,probability=.3),dict(start=.2,probability=.6),dict(start=.45,probability=.8)],
        new_F=600,external_F=528,supervised_PAMF_F=66,native_PAMF_F=6,new_G=4,
        excluded=['external_CE_only','fixed_L1080_V5400','chrono_training','m5','new_fine_ablations'],
        no_target_feedback=True,default_head_joint=True,allowed_gpus=[0,1,2,3,4],reserved_gpus=[5,6,7],max_own_compute=15)
    validate(p);return p

def validate(p):
    if len(p['rows'])!=600 or len(p['reused_rows'])!=27 or len(p['G_rows'])!=4:raise ValueError('Scope mismatch')
    if len(p['protocols'])!=11 or len({r['row_id'] for r in p['rows']+p['reused_rows']})!=627:raise ValueError('Duplicate matrix')
    for pid,q in p['protocols'].items():
        if q['source_days']!=[1,2,3] or set(q['source_rxs'])&set(q['target_rxs']):raise ValueError('Data-role mismatch')
        if q.get('fixed_labels') is not None:raise ValueError('Cancelled fixed budget')
    for r in p['rows']:
        if r['kind']=='external' and (r['variant']!='full' or r['family'] not in FAMILIES):raise ValueError('Only four Full methods')
        if r['model_seed'] not in SEEDS:raise ValueError('Unregistered seed')
    return p
