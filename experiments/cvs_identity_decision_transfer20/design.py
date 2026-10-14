"""Source falsification of default-decision identity residual transfer."""
import math
from pathlib import Path
from experiments.cvs_crossrx_sharpness20 import design as parent
ROOT=Path(__file__).resolve().parents[2]
PROJECT,read,write=parent.PROJECT,parent.read,parent.write
RUN='20261015-phase1-SIGTIME-identity-decision-transfer-sourceL-s2026092701-r01'
RELEASE='cvs_SIGTIME_identity_decision_transfer_sourceL_20261015_r01'
SOURCE_RELEASE=PROJECT+'/releases/'+parent.RELEASE
OUTPUT=PROJECT+'/runs/'+RUN+'/source_diagnostic'
MODES=('ZERO','SAME_RX','CROSS_RX');STATES=('FINAL20',);ORDINALS=(2,19,37,54)
SOURCE_RX=(1,3,4,6,8);LR=2e-4
METHOD=dict(view='z_donor - mean6TX(z_donor) + mean6TX(z_anchor), both side directions, same day',
    mean_scope='current visibleL group only, differentiable, not persistent or deployment normalization',
    readout='original cosine W.detach scale30 eps1e-4, FP64 analytic formula with FP32 comparison',
    stop_probability=.9,weight=.1,loss='weight mean relu(log(.9)-logp_true), 720 synthetic decisions/control',
    gradients='both actualjoint groups including all3gains; no auxiliary W gradient',
    feature_transform_at_deployment=False,new_parameters=0,head_fit_updates=0)
GATE=dict(minimum_qualified_cohorts=2,current_active_decisions_min=36,
    actual_F_difference_norm_min=1e-6,held_feature_RMS_difference_min=1e-5,
    shallow_middle_difference_tensors_min=1,held_transfer_CE_difference_max=-1e-4,
    held_transfer_errors_increase_tolerance=0,held_original_CE_increase_tolerance=1e-6,
    held_original_accuracy_difference_min=0.,controls=['ZERO','SAME_RX'],readouts=['ACTUAL','PRE_W'],
    qualification_transfer_readout='PRE_W: before actual defaultW, afteractualjoint including3gains',
    interpretation='Source availability and compatible default-decision learning; not unseenRX/DG')

def request():
    return dict(run_id=RUN,output_root=OUTPUT,source_release=SOURCE_RELEASE,
        source_row='BASE_CE-s2026092701',model_seed=2026092701,states=list(STATES),modes=list(MODES),
        LR=LR,optimizer_state='saved ownfinal20 m/v/step and original options, newstageLR2e-4',
        current_pairs=[[2,3],[19,20],[37,38],[54,55]],held_ordinals=[66,67],source_role='L_s',
        source_subset='reuse ROWconfirmation900; newmechanism sourceprobe, not independent newdata confirmation',
        method=METHOD,source_falsification=GATE,original_default_head_joint=True,
        feature_scope='actual encoder F71 including3gains; W excluded from auxiliary',
        readouts=['ACTUAL','PRE_W'],PRE_W_gains='after-step actual gains retained',main_loss='original clean CE',
        head_fit_updates=0,U_IQ_read=False,V_IQ_read=False,U_truth_read=False,target_access=False,
        analysis_code_commit='release_commit.txt')

def validate(c):
    expected=request();expected['analysis_code_commit']=c.get('analysis_code_commit')
    if c!=expected or not isinstance(c['analysis_code_commit'],str) or len(c['analysis_code_commit'])!=40:
        raise ValueError('Fixed decision-transfer request or Gitcommit differs')
    return c

def source_gate(records,paired):
    if [(r['state'],r['ordinal'],r['mode']) for r in records]!=[(s,o,m) for s in STATES for o in ORDINALS for m in MODES]:raise ValueError('All12 fixed transfer records required')
    if [(p['state'],p['ordinal']) for p in paired]!=[(s,o) for s in STATES for o in ORDINALS]:raise ValueError('All4 fixed comparisons required')
    qualified=[];details=[]
    for p in paired:
        candidate=next(r for r in records if (r['state'],r['ordinal'],r['mode'])==(p['state'],p['ordinal'],'CROSS_RX'));reasons=[];values=[candidate['auxiliary_gradient_norm']]
        if candidate['auxiliary_gradient_norm']<=0 or candidate['current_transfer']['CROSS_RX']['active_decisions']<GATE['current_active_decisions_min']:reasons.append('insufficient_current_trueidentity_learning_opportunity')
        for control in GATE['controls']:
            q=p['vs_'+control]
            for key,minimum in [('actual_F_difference_norm',GATE['actual_F_difference_norm_min']),('held_feature_RMS_difference',GATE['held_feature_RMS_difference_min'])]:
                values.append(q[key])
                if q[key]<=minimum:reasons.append(control+':'+key)
            if q['shallow_middle_difference_tensors']<GATE['shallow_middle_difference_tensors_min']:reasons.append(control+':no_shallow_middle_change')
            values.append(q['held_transfer_CE_difference'])
            if q['held_transfer_CE_difference']>GATE['held_transfer_CE_difference_max']:reasons.append(control+':no_default_transfer_CE_gain')
            if q['held_transfer_errors_increase']>GATE['held_transfer_errors_increase_tolerance']:reasons.append(control+':transfer_error_harm')
            for readout in GATE['readouts']:
                values.extend([q[readout+'_CE_difference'],q[readout+'_accuracy_difference']])
                if q[readout+'_CE_difference']>GATE['held_original_CE_increase_tolerance']:reasons.append(control+':'+readout+'_CE_harm')
                if q[readout+'_accuracy_difference']<GATE['held_original_accuracy_difference_min']:reasons.append(control+':'+readout+'_accuracy_harm')
        if not all(math.isfinite(float(v)) for v in values):raise ValueError('Nonfinite source transferqualification')
        if not reasons:qualified.append(p['ordinal'])
        details.append(dict(state=p['state'],ordinal=p['ordinal'],qualified=not reasons,failed_conditions=reasons))
    passed=len(qualified)>=GATE['minimum_qualified_cohorts']
    return dict(passed=passed,stage_signal='E20_PLUS20_SIGNAL' if passed else None,qualified_ordinals=qualified,
        cell_conditions=details,controls_not_fallback=True,interpretation=GATE['interpretation'])
