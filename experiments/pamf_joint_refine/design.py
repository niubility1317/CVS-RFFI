import numpy as np
from experiments.pamf_posterior_refine.design import configuration as posterior_configuration
RUN='20261019-phase1-restoration-joint-frozenF-source-s2026092701-m2-r01'
CONTROL_RUN='20261019-phase1-restoration-frozenF-feature-source-s2026092701-m2-r01'
PARENT_RUN='20261019-phase1-restoration-residual-refinement-selfsup-m3-r03'
PARENT_COMMIT='b01b534f52daf9cae9e940ed6ec6f9243bccbdbe'
ARMS=('joint_feature','joint_posterior');ARM='joint_posterior';GPUS=dict(zip(ARMS,(0,1)))
MODEL_SEED=2026092701;DATA_SEED=2026101951;EPOCHS=10;UPDATES=2470;BATCH=256


def configuration(arm):
    if arm not in ARMS:raise ValueError('Registered joint arm required')
    value=posterior_configuration('posterior_feature')
    value.update(schema='joint_frozenF_refine_v1',run_id=RUN,arm=arm,GPU=GPUS[arm],epochs=EPOCHS,updates=UPDATES,posterior_weight=.1 if arm==ARM else 0.,
        trainable='All original R03 coarse andcorrection parameters;F/defaulthead fullyfrozen',
        component_budget='Parentcoarse20/correction20+joint10=coarse30/correction30;60componentepochs,10newpasses,50serialinheritedpasses',
        source_rule='Eacharm vs fixedfeature:lowurban+.5pp andmidurban+.5pp,cleandrop<=.1pp,NMSE<=1.05;selectmaxmeanlow/mid,thenlowerNMSE',
        target_scope='Only source-selected qualifiedjoint arm;automatic168000clean+sixpractical/3originalFheads;no targetreordering',
        matched_comparison='Two joint arms have identical trainableparameters,passes,init,data/order/optimizer;priorcorrection-only is reference,not equalcomponentcost')
    return value


def source_rule(current,control):
    if set(current)!=set(ARMS):raise ValueError('Both fixed joint arms must finish source scoring')
    for item in [*current.values(),control]:
        if not item.get('source_only') or item.get('count')!=1800 or item.get('target_reads')!=0:raise ValueError('Matched source panels only')
    def row(item,view,name='refined'):
        return next(r for r in item['records'] if r['view']==view and r['readout']==name and r['baseline']=='identity' and r['dimension']=='overall')
    control_nmse=float(np.mean([r['NMSE']['refined'] for r in control['waveform'] if r['view']!='clean']))
    records={}
    for arm,item in current.items():
        accuracy={view:row(item,view)['accuracy'] for view in ('practical_low_urban','practical_mid_urban')}
        gains={view:v-row(control,view)['accuracy'] for view,v in accuracy.items()}
        drop=row(item,'clean','identity')['accuracy']-row(item,'clean')['accuracy']
        nmse=float(np.mean([r['NMSE']['refined'] for r in item['waveform'] if r['view']!='clean']))
        finite=all(np.isfinite(v) for v in [*accuracy.values(),*gains.values(),drop,nmse,control_nmse])
        records[arm]=dict(qualified=bool(finite and all(v>=.5 for v in gains.values()) and drop<=.1 and nmse<=1.05*control_nmse),
            accuracy_gains_pp=gains,urban_accuracy_mean=float(np.mean(list(accuracy.values()))),clean_drop_pp=drop,mean_practical_NMSE=nmse)
    eligible=[arm for arm in ARMS if records[arm]['qualified']]
    eligible.sort(key=lambda arm:(-records[arm]['urban_accuracy_mean'],records[arm]['mean_practical_NMSE'],ARMS.index(arm)))
    return dict(qualified=bool(eligible),selected_arm=eligible[0] if eligible else None,candidates=records,
        control_mean_practical_NMSE=control_nmse,selection='Maxmean low/midurban,thenlowerNMSE,thenregisteredarmorder;sourceonly')
