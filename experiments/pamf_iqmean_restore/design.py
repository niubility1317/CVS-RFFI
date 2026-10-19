import numpy as np

RUN='20261020-phase1-restoration-iqmean-source-s2026092701-m2-r01'
BASE_RUN='20261020-phase1-restoration-identity-priority-source-s2026092701-m2-r01'
BASE_COMMIT='de3118c724026869581a9cf1975b0d1314c96d6e'
PARTNER_RUN='20261020-phase1-restoration-layerwise50-source-s2026092701-m2-r01'
PARTNER_COMMIT='672a36de61da32f3974472db97448b8977bbae71'
ARMS=('mean_final','mean_layerwise');PARTNERS=dict(zip(ARMS,('final_guidance','layerwise_guidance')))
GPUS=dict(zip(ARMS,(4,5)));MODEL_SEED=2026092701
F_SEEDS=(2026092701,2026092702,2026092703)
WAVEFORM_REFERENCE_NMSE=.028326367493718863


def configuration(arm):
    if arm not in ARMS:raise ValueError('Registered IQ-mean arm required')
    return dict(schema='fixed_frozen_iqmean_v1',run_id=RUN,arm=arm,weights=[.5,.5],
        members=[dict(run=BASE_RUN,commit=BASE_COMMIT,arm='posterior03',epoch=10,updates=2470),
                 dict(run=PARTNER_RUN,commit=PARTNER_COMMIT,arm=PARTNERS[arm],epoch=30,updates=7410)],
        operation='Arithmetic mean of two aligned IQ outputs on the same received record; no logits or coordinate mean',
        parameter_updates=0,new_training_epochs=0,head_fit_updates=0,F_optimizer_updates=0,TX_labels_read=0,
        member_training_seed=MODEL_SEED,source_count=1800,source_role='V',F_seeds=list(F_SEEDS),
        inherited_member_component_epochs=[60,80],unique_inherited_component_epochs=120,
        inherited_member_serial_passes=[50,50],inherited_shared_original_coarse_epochs=20,
        new_trainable_parameters=0,inference_member_count=2,inference_parameters_expected=4281680,
        source_rule='vs posterior03: each urban+.5pp;otherfour>=-.5pp;clean drop<=.1pp;NMSE<=1.15 original feature',
        selection='Highest source urban mean,lower NMSE,registered order;selected arm only receives target',
        GPU=GPUS[arm],all_owner_cap_per_GPU=2,min_free_MB=12000,CPU_threads=2,
        target_scope='All168000 clean+six practical;3 independent original defaultheads;all seven before CPU truth',
        target_training=False,paired_clean_in_practical_inference=False,state_updates=0,
        limitation='New composed candidate; failed members remain failed. Extra inference cost is reported, not free training.')


def source_rule(current,control):
    if set(current)!=set(ARMS):raise ValueError('Both registered source arms must complete')
    for item in [*current.values(),control]:
        if not item.get('source_only') or item.get('count')!=1800 or item.get('target_reads')!=0:
            raise ValueError('Matching source-only panels required')
    def row(item,view,name='refined'):
        return next(r for r in item['records'] if r['view']==view and r['readout']==name and r['baseline']=='identity' and r['dimension']=='overall')
    urban=('practical_low_urban','practical_mid_urban')
    others=('practical_high','practical_mid','practical_low_suburban','practical_high_urban');records={}
    for arm,item in current.items():
        accuracy={v:row(item,v)['accuracy'] for v in urban}
        gains={v:accuracy[v]-row(control,v)['accuracy'] for v in urban}
        other={v:row(item,v)['accuracy']-row(control,v)['accuracy'] for v in others}
        drop=row(item,'clean','identity')['accuracy']-row(item,'clean')['accuracy']
        nmse=float(np.mean([r['NMSE']['refined'] for r in item['waveform'] if r['view']!='clean']))
        finite=all(np.isfinite(x) for x in [*gains.values(),*other.values(),drop,nmse])
        records[arm]=dict(qualified=bool(finite and min(gains.values())>=.5 and min(other.values())>=-.5 and
            drop<=.1 and nmse<=1.15*WAVEFORM_REFERENCE_NMSE),accuracy_gains_pp=gains,other_accuracy_gains_pp=other,
            urban_accuracy_mean=float(np.mean(list(accuracy.values()))),clean_drop_pp=drop,mean_practical_NMSE=nmse)
    eligible=[a for a in ARMS if records[a]['qualified']]
    eligible.sort(key=lambda a:(-records[a]['urban_accuracy_mean'],records[a]['mean_practical_NMSE'],ARMS.index(a)))
    return dict(qualified=bool(eligible),selected_arm=eligible[0] if eligible else None,candidates=records,
        control_run=BASE_RUN,waveform_reference_NMSE=WAVEFORM_REFERENCE_NMSE,no_target_feedback=True)
