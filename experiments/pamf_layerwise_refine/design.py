import numpy as np

RUN='20261020-phase1-restoration-layerwise50-source-s2026092701-m2-r01'
PARENT_RUN='20261019-phase1-reception-restoration-selfsup-manysig-m6-r02'
PARENT_COMMIT='7efdba92932a86bf4b0492a565d35377564a26da'
CALIBRATION_RUN='20261019-phase1-restoration-residual-refinement-selfsup-m3-r03'
CONTROL_RUN='20261020-phase1-restoration-identity-priority-source-s2026092701-m2-r01'
PRECEDING_RUN=CONTROL_RUN
ARMS=('final_guidance','layerwise_guidance');ARM=ARMS[1];GPUS=dict(zip(ARMS,(4,5)))
TEACHER_SEEDS=(2026092701,2026092702,2026092703)
MODEL_SEED=2026092701;DATA_SEED=2026101951;EPOCHS=30;UPDATES=7410;BATCH=256;CACHE_EPOCHS=20
WAVEFORM_REFERENCE_NMSE=.028326367493718863


def cache_epoch(epoch):
    if not isinstance(epoch,int) or not 1<=epoch<=EPOCHS:raise ValueError('Registered epoch required')
    return (epoch-1)%CACHE_EPOCHS+1


def configuration(arm):
    if arm not in ARMS:raise ValueError('Registered layerwise arm required')
    return dict(schema='layerwise_frozenF_refine_v1',run_id=RUN,arm=arm,parent_run=PARENT_RUN,parent_commit=PARENT_COMMIT,
        initialization='Direct original R02 coarse20; new zero-output correction; no R03 or refined correction weights',
        calibration_source=CALIBRATION_RUN+' source-only pre-optimizer scalar; not a trained model inheritance',
        model_seed=MODEL_SEED,data_seed=DATA_SEED,epochs=EPOCHS,updates=UPDATES,physical_batch=BATCH,
        IQ_examples_per_physical=2,source_count=63000,L_count=6300,U_count=56700,
        routes='Original R02 .15 clean/.60 residual/.25 full; observed->clean and clean->clean equally weighted',
        cache_schedule='Epochs1..20 then1..10 repeated; no new IQ generation',optimizer='AdamW reset',weight_decay=1e-4,
        coarse_learning_rate=3e-5,correction_learning_rate=2e-4,warmup_updates=247,final_lr_ratio=.1,clip_grad_norm=1.,precision='FP32',
        teacher_seeds=list(TEACHER_SEEDS),teacher_aggregation='mean of per-teacher scalar losses; no coordinate averaging',
        feature_weight=1.,posterior_weight=.3,posterior_temperature=2.,middle_weight=.1 if arm==ARM else 0.,middle_layers=['t1','t2'],
        loss='IQ NMSE+.1logpower+mean_teacher(final cosine+.3T2 squared KL+middle_weight*mean(t1,t2 cosine))',
        middle_reference='Source paired clean computed no_grad only in layered arm; original teacher features/heads frozen',
        TX_labels_read=0,target_training=False,F_optimizer_updates=0,head_fit_updates=0,coarse_updates=UPDATES,
        parent_coarse_epochs=20,parent_corrector_epochs=0,total_coarse_epochs=50,total_correction_epochs=30,
        component_epochs=80,serial_inherited_passes=50,checkpoint_selection='Fixed finalE30; no V optimization or earlystop',
        control_run=CONTROL_RUN,control_arm='posterior03',control_reuse='Completed source E10 metrics only; not initialization or equal cost claim',
        source_readout='Same fixed F1 source V1800/clean+six practical; all predictions before source truth',
        source_rule='vs source posterior03: each urban+.5pp; otherfour>=-.5pp; clean drop<=.1pp; NMSE<=1.15 ORIGINAL feature reference',
        target_scope='Only source-selected qualified arm; all168000 clean+six practical/3 independent original default heads; truth last',
        GPU=GPUS[arm],max_this_arm_GPU_workers=1,all_owner_cap_per_GPU=2,min_free_MB=18000,target_min_free_MB=12000,
        capacity_lock='receiver_residual_capacity4.lock',CPU_threads=2,
        stop_rule='Illegal source/parent/code,collision,nonfinite,budget/prediction failure: preserve outputs,no automatic retry',
        matched_comparison='New arms match init/parameters/E30/data/order/optimizer; middle forward compute differs and is measured')


def source_rule(current,control):
    if set(current)!=set(ARMS):raise ValueError('Both registered source arms must complete')
    for item in [*current.values(),control]:
        if not item.get('source_only') or item.get('count')!=1800 or item.get('target_reads')!=0:
            raise ValueError('Fixed legal source panels required')
    def row(item,view,name='refined'):
        return next(r for r in item['records'] if r['view']==view and r['readout']==name and r['baseline']=='identity' and r['dimension']=='overall')
    urban=('practical_low_urban','practical_mid_urban')
    others=('practical_high','practical_mid','practical_low_suburban','practical_high_urban');records={}
    for arm,item in current.items():
        accuracy={v:row(item,v)['accuracy'] for v in urban};gains={v:accuracy[v]-row(control,v)['accuracy'] for v in urban}
        other_gains={v:row(item,v)['accuracy']-row(control,v)['accuracy'] for v in others}
        drop=row(item,'clean','identity')['accuracy']-row(item,'clean')['accuracy']
        nmse=float(np.mean([r['NMSE']['refined'] for r in item['waveform'] if r['view']!='clean']))
        finite=all(np.isfinite(x) for x in [*gains.values(),*other_gains.values(),drop,nmse])
        records[arm]=dict(qualified=bool(finite and min(gains.values())>=.5 and min(other_gains.values())>=-.5 and
            drop<=.1 and nmse<=1.15*WAVEFORM_REFERENCE_NMSE),accuracy_gains_pp=gains,other_accuracy_gains_pp=other_gains,
            urban_accuracy_mean=float(np.mean(list(accuracy.values()))),clean_drop_pp=drop,mean_practical_NMSE=nmse)
    eligible=[a for a in ARMS if records[a]['qualified']]
    eligible.sort(key=lambda a:(-records[a]['urban_accuracy_mean'],records[a]['mean_practical_NMSE'],ARMS.index(a)))
    return dict(qualified=bool(eligible),selected_arm=eligible[0] if eligible else None,candidates=records,
        control_run=CONTROL_RUN,waveform_reference_NMSE=WAVEFORM_REFERENCE_NMSE,
        selection='Maximum mean source low/mid urban, then lower NMSE, then registered order; no target feedback')
