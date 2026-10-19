import numpy as np
from experiments.pamf_identity_priority.design import configuration as base_configuration
from experiments.pamf_identity_priority.design import PARENT_RUN,PARENT_COMMIT,TEACHER_SEEDS,MODEL_SEED,DATA_SEED,EPOCHS,UPDATES,BATCH,WAVEFORM_REFERENCE_NMSE

RUN='20261020-phase1-restoration-hardness-source-s2026092701-m3-r01'
CONTROL_RUN='20261020-phase1-restoration-identity-priority-source-s2026092701-m2-r01'
PRECEDING_RUN='20261020-phase1-restoration-iqmean-source-s2026092701-m2-r01'
ARMS=('mean_observed','cvar_observed','rxdro_observed');ARM=ARMS[0]
GPUS=dict(zip(ARMS,(4,5,6)));RX=(1,3,4,6,8)


def configuration(arm):
    if arm not in ARMS:raise ValueError('Registered hardness arm required')
    value=base_configuration('posterior03')
    value.update(schema='hardness_frozenF_refine_v1',run_id=RUN,arm=arm,GPU=GPUS[arm],
        control_run=CONTROL_RUN,control_arm='posterior03',posterior_weight=.3,teacher_aggregation='mean',
        focus_weight=.5,focus_fraction=.25,rxdro_eta=.01,source_RX=list(RX),
        focus_method=dict(zip(ARMS,('mean observed semantic loss','top25percent observed semantic loss','adaptive RX group mean observed semantic loss')))[arm],
        loss='mean_all_pairs(IQNMSE+.1logpower+semantic)+.5*focus(semantic_observed);semantic=mean3F(finalcos+.3T2KL)',
        rxdro='Train-only logq+=.01*detached per-RX observed semantic mean;softmax over present groups;uniform initialization',
        clean_preservation='Original observed+clean pairs equally weighted in base loss;additional focus uses observed half only',
        RX_use='Legal train/rx.npy only;no TX truth,receiver classifier or label-derived hard sample selection',
        focus_state='Cumulative actual selected exposures/RX counts and final train-only weights preserved;no runtime adaptation',
        checkpoint_selection='Fixed finalE10,2470 updates;no sourceV updates or earlystop',
        control_reuse='Completed posterior03 source metrics only;new mean_observed is matched control for extra observed term',
        source_rule='vs completed posterior03:each urban+.5pp,otherfour>=-.5pp,clean drop<=.1pp,NMSE<=1.15 original feature',
        matched_comparison='Three new arms match originalR03 init/parameters/E10/data/order/routes/optimizer/3F;only additional observed loss aggregation differs',
        target_scope='Only source-selected qualified arm;168000 clean+six practical,3 independent original default heads,all7 before CPU truth')
    return value


def source_rule(current,control):
    if set(current)!=set(ARMS):raise ValueError('All registered source arms must complete')
    for item in [*current.values(),control]:
        if not item.get('source_only') or item.get('count')!=1800 or item.get('target_reads')!=0:
            raise ValueError('Fixed legal source panels required')
    def row(item,view,name='refined'):
        return next(r for r in item['records'] if r['view']==view and r['readout']==name and r['baseline']=='identity' and r['dimension']=='overall')
    urban=('practical_low_urban','practical_mid_urban')
    others=('practical_high','practical_mid','practical_low_suburban','practical_high_urban')
    records={}
    for arm,item in current.items():
        accuracy={v:row(item,v)['accuracy'] for v in urban}
        gains={v:accuracy[v]-row(control,v)['accuracy'] for v in urban}
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
