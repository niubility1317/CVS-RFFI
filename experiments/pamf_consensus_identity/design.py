import numpy as np
from experiments.pamf_identity_priority.design import configuration as base_configuration
from experiments.pamf_identity_priority.design import PARENT_RUN,PARENT_COMMIT,TEACHER_SEEDS,MODEL_SEED,DATA_SEED,EPOCHS,UPDATES,BATCH,WAVEFORM_REFERENCE_NMSE

RUN='20261020-phase1-restoration-consensus-identity-source-s2026092701-m2-r01'
CONTROL_RUN='20261020-phase1-restoration-identity-priority-source-s2026092701-m2-r01'
PRECEDING_RUN='20261020-phase1-restoration-hardness-source-s2026092701-m3-r01'
ARMS=('consensus_native','consensus_crossrx');ARM=ARMS[0]
GPUS=dict(zip(ARMS,(4,5)));RX=(1,3,4,6,8)
CONFIDENCE=.9;NATIVE_WEIGHT=.3;PROTOTYPE_WEIGHT=.3


def configuration(arm):
    if arm not in ARMS:raise ValueError('Registered consensus identity arm required')
    value=base_configuration('posterior03')
    value.update(schema='consensus_identity_refine_v1',run_id=RUN,arm=arm,GPU=GPUS[arm],
        control_run=CONTROL_RUN,control_arm='posterior03',posterior_weight=.3,teacher_aggregation='mean',
        source_RX=list(RX),consensus_confidence=CONFIDENCE,native_weight=NATIVE_WEIGHT,
        prototype_weight=PROTOTYPE_WEIGHT if arm==ARMS[1] else 0.,
        consensus='All3 original defaultheads agree on clean source class;each native T1 probability>=.9;detached pseudo identity only',
        prototype='Per teacher separate unit-feature means by pseudo class/RX;equal means of available other RX;unit normalize;exclude own RX',
        loss='mean_all_pairs(IQNMSE+.1logpower+mean3F(finalcos+.3T2KL))+.3*mean_observed(masked native CE)+prototype_weight*mean_observed(masked otherRX cosine)',
        loss_denominator='All observed records,including rejected records as zero;no resampling;original paired losses see all records',
        train_only_bank='Source L/U clean features/pseudo identities/RX,computed once without TX truth;not a fitted head;not used for inference',
        checkpoint_selection='Fixed finalE10,2470 updates;no sourceV updates or earlystop',
        matched_comparison='Same originalR03/init/E10/parameters/data/order/routes/optimizer/threeF/consensus CE;only additional otherRX feature loss differs',
        source_rule='Prospective low-first:vs posterior03 low>=+.5pp,mid>=0pp,otherfour>=-.5pp,clean drop<=.1pp,NMSE<=1.15 original feature;no retrospective qualification',
        target_scope='Only source-selected qualified arm;168000 clean+six practical,three independent original defaultheads,all7 before CPU truth')
    return value


def source_rule(current,control):
    if set(current)!=set(ARMS):raise ValueError('All registered source arms must complete')
    for item in [*current.values(),control]:
        if not item.get('source_only') or item.get('count')!=1800 or item.get('target_reads')!=0:
            raise ValueError('Fixed legal source panels required')
    def row(item,view,name='refined'):
        return next(r for r in item['records'] if r['view']==view and r['readout']==name and r['baseline']=='identity' and r['dimension']=='overall')
    urban=('practical_low_urban','practical_mid_urban')
    others=('practical_high','practical_mid','practical_low_suburban','practical_high_urban');records={}
    for arm,item in current.items():
        accuracy={v:row(item,v)['accuracy'] for v in urban}
        gains={v:accuracy[v]-row(control,v)['accuracy'] for v in urban}
        other_gains={v:row(item,v)['accuracy']-row(control,v)['accuracy'] for v in others}
        drop=row(item,'clean','identity')['accuracy']-row(item,'clean')['accuracy']
        nmse=float(np.mean([r['NMSE']['refined'] for r in item['waveform'] if r['view']!='clean']))
        finite=all(np.isfinite(x) for x in [*gains.values(),*other_gains.values(),drop,nmse])
        records[arm]=dict(qualified=bool(finite and gains[urban[0]]>=.5-1e-8 and gains[urban[1]]>=-1e-8 and
            min(other_gains.values())>=-.5-1e-8 and drop<=.1+1e-8 and nmse<=1.15*WAVEFORM_REFERENCE_NMSE),
            accuracy_gains_pp=gains,other_accuracy_gains_pp=other_gains,urban_accuracy=accuracy,
            clean_drop_pp=drop,mean_practical_NMSE=nmse)
    eligible=[a for a in ARMS if records[a]['qualified']]
    eligible.sort(key=lambda a:(-records[a]['urban_accuracy'][urban[0]],-records[a]['urban_accuracy'][urban[1]],
                                records[a]['mean_practical_NMSE'],ARMS.index(a)))
    return dict(qualified=bool(eligible),selected_arm=eligible[0] if eligible else None,candidates=records,
        control_run=CONTROL_RUN,waveform_reference_NMSE=WAVEFORM_REFERENCE_NMSE,
        selection='Source low-urban highest,then mid-urban,lower NMSE,registered order;no target feedback')
