import numpy as np
from experiments.pamf_identity_priority.design import configuration as base_configuration
from experiments.pamf_identity_priority.design import PARENT_RUN,PARENT_COMMIT,TEACHER_SEEDS,MODEL_SEED,DATA_SEED,EPOCHS,UPDATES,BATCH,WAVEFORM_REFERENCE_NMSE

RUN='20261020-phase1-restoration-semantic-feedback-source-s2026092701-m2-r01'
CONTROL_RUN='20261020-phase1-restoration-identity-priority-source-s2026092701-m2-r01'
PRECEDING_RUN='20261020-phase1-restoration-consensus-identity-source-s2026092701-m2-r01'
ARMS=('iq_condition','semantic_condition');ARM=ARMS[0]
GPUS=dict(zip(ARMS,(4,5)));FEEDBACK_SEED=TEACHER_SEEDS[0]
MODULATION_BOUND=.1;MODULATION_HIDDEN=128;TRAINABLE_PARAMETERS=2244008

def configuration(arm):
    if arm not in ARMS:raise ValueError('Registered feedback arm required')
    value=base_configuration('posterior03')
    value.update(schema='semantic_feedback_refine_v1',run_id=RUN,arm=arm,GPU=GPUS[arm],
        control_run=CONTROL_RUN,control_arm='posterior03',posterior_weight=.3,teacher_aggregation='mean',
        condition='unit normalized current received+coarse IQ adaptiveavgpool4x40' if arm==ARMS[0] else 'unit normalized original frozen F1 features of current coarse IQ',
        runtime_F_seed=None if arm==ARMS[0] else FEEDBACK_SEED,modulation_bound=MODULATION_BOUND,
        modulation='160->128 SiLU->640;zero final layer;per scale gamma/beta bounded by.1tanh;applied to all four correction encoder/skip levels',
        trainable_parameters=TRAINABLE_PARAMETERS,modulation_LR='2e-4,warm247,cos to2e-5',legacy_LR='3e-5,warm247,cos to3e-6',
        loss='Original posterior03:mean_all_pairs(IQNMSE+.1logpower+mean3F(finalcos+.3T2KL));no extra CE/prototype/focus',
        runtime='One current IQ only;no clean/RX/bank/pseudolabels,head fitting,gradient optimization or cross-example state at inference',
        runtime_cost='Semantic restorer carries frozen original F1 and adds feature pass;F1 instance is shared with scoring when available;standalone state differs,actual walltime/peak reported',
        matched_comparison='Same originalR03/zeroFiLM/init/E10/trainableparameters/data/order/routes/optimizer/base3Floss;only conditioning code differs',
        checkpoint_selection='Fixed finalE10/2470,no V updates or earlystop',
        source_rule='vs posterior03:low>=+.5pp,mid>=0pp,otherfour>=-.5pp,clean drop<=.1pp,NMSE<=1.15 originalfeature;low-first,no retrospective qualification',
        target_scope='Only source-selected qualified arm,168000 clean+sixpractical,three native heads independentargmax,all7 before CPUtruth')
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
