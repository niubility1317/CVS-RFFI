"""Fixed source assay: learn only receiver-transfer excess identity risk."""
import math
from pathlib import Path
from experiments.cvs_crossrx_sharpness20 import design as parent

ROOT=Path(__file__).resolve().parents[2]
PROJECT,read,write=parent.PROJECT,parent.read,parent.write
RUN='20261015-phase1-SIGTIME-identity-excess-transfer-sourceL-s2026092701-r01'
RELEASE='cvs_SIGTIME_identity_excess_transfer_sourceL_20261015_r01'
SOURCE_RELEASE=PROJECT+'/releases/'+parent.RELEASE
OUTPUT=PROJECT+'/runs/'+RUN+'/source_diagnostic'
MODES=('ZERO','SAME_RX','CROSS_RX');STATES=('INITIAL','FINAL20');ORDINALS=(2,19,37,54)
SOURCE_RX=(1,3,4,6,8);LR=2e-4
METHOD=dict(view='a_y-a_c+b_c and b_y-b_c+a_c, all5 c!=y, same day and opposite sides',
    factual='actual b_y for forward; actual a_y for reverse; same true identity y',
    task='Repair only additional trueclass risk beyond actual destination, not absolute confidence',
    readout='original cosine W.detach scale30 eps1e-4, FP64 formula and FP32 comparison',
    stop_probability=None,tolerance=1e-6,weight=.1,
    loss='.1 mean relu(CE_virtual-CE_factual.detach()-1e-6), fixed3600 denominator/control',
    gradients='virtual all3 actualjoint endpoints and all3gains; factual threshold and W stopped',
    virtual_decisions_per_control=3600,factual_decisions_per_control=3600,
    formula_decisions_per_call=14400,extra_IQ_forwards=0,
    feature_transform_at_deployment=False,new_parameters=0,head_fit_updates=0)
GATE=dict(minimum_qualified_cohorts_per_state=2,current_active_decisions_min=180,
    current_active_origin_packets_min=18,absolute_auxiliary_F_cosine_main_max=.95,
    actual_F_difference_norm_min=1e-6,held_feature_RMS_difference_min=1e-5,
    shallow_middle_difference_tensors_min=1,
    held_positive_excess_risk_difference_max=-1e-4,held_transfer_CE_difference_max=-1e-4,
    held_transfer_errors_increase_tolerance=0,held_original_CE_relative_increase_tolerance=.01,
    held_original_accuracy_difference_min=0.,controls=['ZERO','SAME_RX'],readouts=['ACTUAL','PRE_W'],
    qualification_transfer_readout='PRE_W before defaultW, same afteractualjoint including3gains',
    original_CE_guard='relative1percent of samecontrol CE, fixed newcandidate only',
    stage_priority='FINAL20 pass prefers sameparent E20+20; only INITIAL pass supports scratch20',
    interpretation='Local source excess-risk and identity gain with distinct gradient, not fulltraining or unseenRX/DG')

def request():
    return dict(run_id=RUN,output_root=OUTPUT,source_release=SOURCE_RELEASE,
        source_row='BASE_CE-s2026092701',model_seed=2026092701,states=list(STATES),modes=list(MODES),
        LR=LR,optimizer_state='INITIAL empty originalmoments; FINAL20 savedm/v/step; both fixedstageLR2e-4',
        current_pairs=[[2,3],[19,20],[37,38],[54,55]],held_ordinals=[66,67],source_role='L_s',
        source_subset='reuse prior900L; different loss, not independent newdata confirmation',
        method=METHOD,source_falsification=GATE,original_default_head_joint=True,
        feature_scope='actual F71 including3gains; W excluded from auxiliary',
        readouts=['ACTUAL','PRE_W'],PRE_W_gains='after-step actual gains retained',main_loss='original clean CE',
        head_fit_updates=0,U_IQ_read=False,V_IQ_read=False,U_truth_read=False,target_access=False,
        analysis_code_commit='release_commit.txt')

def validate(c):
    expected=request();expected['analysis_code_commit']=c.get('analysis_code_commit')
    if c!=expected or not isinstance(c['analysis_code_commit'],str) or len(c['analysis_code_commit'])!=40:
        raise ValueError('Fixed excess-transfer request or Gitcommit differs')
    return c

def source_gate(records,paired):
    if [(r['state'],r['ordinal'],r['mode']) for r in records]!=[(s,o,m) for s in STATES for o in ORDINALS for m in MODES]:raise ValueError('All24 fixed excess records required')
    if [(p['state'],p['ordinal']) for p in paired]!=[(s,o) for s in STATES for o in ORDINALS]:raise ValueError('All8 fixed comparisons required')
    qualified={s:[] for s in STATES};details=[]
    for p in paired:
        candidate=next(r for r in records if (r['state'],r['ordinal'],r['mode'])==(p['state'],p['ordinal'],'CROSS_RX'));reasons=[]
        cosine=candidate['auxiliary_F_gradient_cosine_main'];values=[candidate['auxiliary_gradient_norm']]
        current=candidate['current_transfer']['CROSS_RX']
        if candidate['auxiliary_gradient_norm']<=0 or current['active_decisions']<GATE['current_active_decisions_min'] or current['active_origin_packet_count']<GATE['current_active_origin_packets_min']:reasons.append('insufficient_current_distinct_identity_opportunity')
        if cosine is None:reasons.append('no_distinct_F_gradient')
        else:
            values.append(cosine)
            if abs(cosine)>GATE['absolute_auxiliary_F_cosine_main_max']:reasons.append('near_plain_CE_gradient_amplification')
        for control in GATE['controls']:
            q=p['vs_'+control]
            for key,minimum in [('actual_F_difference_norm',GATE['actual_F_difference_norm_min']),('held_feature_RMS_difference',GATE['held_feature_RMS_difference_min'])]:
                values.append(q[key])
                if q[key]<=minimum:reasons.append(control+':'+key)
            if q['shallow_middle_difference_tensors']<GATE['shallow_middle_difference_tensors_min']:reasons.append(control+':no_shallow_middle_change')
            for key in ('held_positive_excess_risk_difference','held_transfer_CE_difference'):
                values.append(q[key])
                if q[key]>GATE[key+'_max']:reasons.append(control+':no_'+key+'_gain')
            if q['held_transfer_errors_increase']>GATE['held_transfer_errors_increase_tolerance']:reasons.append(control+':transfer_error_harm')
            for readout in GATE['readouts']:
                values.extend([q[readout+'_CE_relative_difference'],q[readout+'_accuracy_difference']])
                if q[readout+'_CE_relative_difference']>GATE['held_original_CE_relative_increase_tolerance']:reasons.append(control+':'+readout+'_relative_CE_harm')
                if q[readout+'_accuracy_difference']<GATE['held_original_accuracy_difference_min']:reasons.append(control+':'+readout+'_accuracy_harm')
        if not all(math.isfinite(float(v)) for v in values):raise ValueError('Nonfinite source excess qualification')
        if not reasons:qualified[p['state']].append(p['ordinal'])
        details.append(dict(state=p['state'],ordinal=p['ordinal'],qualified=not reasons,failed_conditions=reasons))
    minimum=GATE['minimum_qualified_cohorts_per_state']
    stage='E20_PLUS20_SIGNAL' if len(qualified['FINAL20'])>=minimum else 'SCRATCH20_SIGNAL' if len(qualified['INITIAL'])>=minimum else None
    return dict(passed=stage is not None,stage_signal=stage,qualified_ordinals_by_state=qualified,
        cell_conditions=details,controls_not_fallback=True,interpretation=GATE['interpretation'])
