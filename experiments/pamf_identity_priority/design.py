import numpy as np
from experiments.pamf_posterior_refine.design import configuration as parent_configuration

RUN='20261020-phase1-restoration-identity-priority-source-s2026092701-m2-r01'
CONTROL_RUN='20261019-phase1-restoration-joint-frozenF-source-s2026092701-m2-r01'
PARENT_RUN='20261019-phase1-restoration-residual-refinement-selfsup-m3-r03'
PARENT_COMMIT='b01b534f52daf9cae9e940ed6ec6f9243bccbdbe'
ARMS=('posterior03','posterior10');ARM=ARMS[0];GPUS=dict(zip(ARMS,(4,5)))
TEACHER_SEEDS=(2026092701,2026092702,2026092703)
PRECEDING_RUN='20261020-phase1-restoration-multireader-source-s2026092701-m2-r01'
MODEL_SEED=2026092701;DATA_SEED=2026101951;EPOCHS=10;UPDATES=2470;BATCH=256
# Original correction-only feature source reference; do not compound the 15% bound.
WAVEFORM_REFERENCE_NMSE=.028326367493718863


def configuration(arm):
    if arm not in ARMS:raise ValueError('Registered identity-priority arm required')
    value=parent_configuration('posterior_feature')
    value.update(schema='identity_priority_frozenF_refine_v1',run_id=RUN,arm=arm,GPU=GPUS[arm],epochs=EPOCHS,updates=UPDATES,
        posterior_weight=.3 if arm==ARMS[0] else 1.,teacher_seeds=list(TEACHER_SEEDS),teacher_aggregation='mean',
        teacher='Original F50 seeds 2026092701/02/03 and their original default heads frozen; no TX truth for restorer',
        feature_loss='Per-teacher 1-cos(estimated embedding, detached same-teacher clean embedding); posterior term recorded separately',
        coarse_updates=UPDATES,checkpoint_selection='Fixed final E10; no V earlystop or V parameter/normalization updates',
        min_free_MB=18000,target_min_free_MB=12000,control_run=CONTROL_RUN,control_arm='joint_posterior',
        control_reuse='Completed matched single-reader E10/2470 source result; no repeated control training',
        trainable='Original R03 coarse and correction; all three original F/default heads frozen',
        component_budget='Parent coarse20/correction20 + joint10 = coarse30/correction30;60 component epochs,50 inherited serial passes',
        source_rule='vs completed single-reader E10: low/mid urban each +.5pp; clean drop <=.1pp; other four views drop <=.5pp; NMSE <=1.15 original feature reference',
        source_readout='Same fixed F1/source V as the completed single-reader control; three F used only as frozen guidance',
        target_scope='Only selected qualified arm; 168000 clean + six practical; three original heads scored independently',
        matched_comparison='Same R03 initialization, trainable parameters, E10, data/order/routes/optimizer; Three teachers fixed; posterior weight .3 versus 1.0 varies, compute disclosed')
    return value


def source_rule(current,control):
    if set(current)!=set(ARMS):raise ValueError('Both registered source arms must complete')
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
