"""One fixed source ablation; reuse the completed equal-budget feature control."""
import numpy as np
from experiments.pamf_feature_refine.design import configuration as feature_configuration
RUN='20261019-phase1-restoration-frozenF-posterior-source-s2026092701-r01'
CONTROL_RUN='20261019-phase1-restoration-frozenF-feature-source-s2026092701-m2-r01'
PARENT_RUN='20261019-phase1-restoration-residual-refinement-selfsup-m3-r03'
PARENT_COMMIT='b01b534f52daf9cae9e940ed6ec6f9243bccbdbe'
ARM='posterior_feature';ARMS=(ARM,);GPUS={ARM:0}
MODEL_SEED=2026092701;DATA_SEED=2026101951;EPOCHS=20;BATCH=256;UPDATES=4940


def configuration(arm):
    if arm!=ARM:raise ValueError('Registered posterior arm required')
    value=feature_configuration('frozenF_feature')
    value.update(schema='frozenF_posterior_refine_v1',run_id=RUN,arm=arm,
        feature_weight=1.,posterior_weight=1.,posterior_temperature=2.,
        posterior_loss='4*KL(softmax(detached nativecleanlogits/2)||softmax(nativeestimatedlogits/2));perpacket',
        teacher='Original F50 seed2026092701/features andoriginaldefaulthead frozen;no TXtruth forrestorer',
        control_run=CONTROL_RUN,control_arm='frozenF_feature',control_reuse='CompletedoriginalequalE20/4940;no retraining',
        source_rule='lowurban andmidurban each>=featurecontrol+.5pp;cleandropvsraw<=.1pp;meanpracticalNMSE<=1.05featurecontrol',
        target_scope='Ifsourcequalifies,automaticallyall168000clean+sixpractical,all3originalFheads,independenttruthlast',
        GPU=0,only_final_checkpoint=True,
        benchmark_context='Targetbenchmark previouslyexposed;newlossandhyperparametersfixedfromsourcefeatureobjective analysis,notnewblindtest')
    return value


def source_rule(current,control):
    if (not current.get('source_only') or not control.get('source_only') or
        current.get('count')!=1800 or control.get('count')!=1800 or
        current.get('target_reads')!=0 or control.get('target_reads')!=0):raise ValueError('Matched completed source panels required')
    def row(item,view,readout='refined'):
        return next(r for r in item['records'] if r['view']==view and r['readout']==readout and r['baseline']=='identity' and r['dimension']=='overall')
    gains={view:row(current,view)['accuracy']-row(control,view)['accuracy'] for view in ('practical_low_urban','practical_mid_urban')}
    clean_drop=row(current,'clean','identity')['accuracy']-row(current,'clean')['accuracy']
    nmse={key:float(np.mean([r['NMSE']['refined'] for r in item['waveform'] if r['view']!='clean'])) for key,item in (('current',current),('control',control))}
    finite=all(np.isfinite(v) for v in [*gains.values(),clean_drop,*nmse.values()])
    qualified=finite and all(v>=.5 for v in gains.values()) and clean_drop<=.1 and nmse['current']<=1.05*nmse['control']
    return dict(qualified=bool(qualified),accuracy_gains_pp=gains,clean_drop_pp=clean_drop,mean_practical_NMSE=nmse)
