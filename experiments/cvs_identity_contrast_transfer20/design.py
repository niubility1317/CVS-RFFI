"""Fixed early/late source probe of cross-identity receiver-change transfer."""
import math
from pathlib import Path
from experiments.cvs_crossrx_sharpness20 import design as parent
ROOT=Path(__file__).resolve().parents[2]
PROJECT,read,write=parent.PROJECT,parent.read,parent.write
RUN='20261015-phase1-SIGTIME-identity-contrast-transfer-sourceL-s2026092701-r01'
RELEASE='cvs_SIGTIME_identity_contrast_transfer_sourceL_20261015_r01'
SOURCE_RELEASE=PROJECT+'/releases/'+parent.RELEASE
OUTPUT=PROJECT+'/runs/'+RUN+'/source_diagnostic'
MODES=('ZERO','SAME_RX','CROSS_RX');STATES=('INITIAL','FINAL20');ORDINALS=(2,19,37,54)
SOURCE_RX=(1,3,4,6,8);LR=2e-4
METHOD=dict(view='a_y-a_c+b_c and b_y-b_c+a_c, every c!=y, same day and opposite sides',
    task='Receiver change measured with another trueTX reference must retain origin trueidentity under original defaultcos',
    readout='original cosine W.detach scale30 eps1e-4, FP64 analytic formula with FP32 comparison',
    stop_probability=.9,weight=.1,loss='weight mean relu(log(.9)-logp_true), 3600 synthetic decisions/control',
    reference_classes='all5 different trueTX; no selection or averaging before defaultreadout',
    gradients='all3 actualjoint endpoints and all3gains; auxiliary W absent',
    feature_transform_at_deployment=False,new_parameters=0,head_fit_updates=0)
GATE=dict(minimum_qualified_cohorts_per_state=2,current_active_decisions_min=180,
    current_active_origin_packets_min=18,actual_F_difference_norm_min=1e-6,
    held_feature_RMS_difference_min=1e-5,shallow_middle_difference_tensors_min=1,
    held_transfer_CE_difference_max=-1e-4,held_transfer_errors_increase_tolerance=0,
    held_original_CE_relative_increase_tolerance=.01,held_original_accuracy_difference_min=0.,
    controls=['ZERO','SAME_RX'],readouts=['ACTUAL','PRE_W'],
    original_CE_guard='relative1percent of samecontrol CE, not old absolute1e-6 qualification; fixed before this candidate results',
    qualification_transfer_readout='PRE_W before defaultW, afteractualjoint including3gains',
    stage_priority='INITIAL pass prefers scratch20; only FINAL20 pass supports sameparent E20+20',
    interpretation='Source opportunity and compatible functional identity transfer, not fulltraining or unseenRX/DG')

def request():
    return dict(run_id=RUN,output_root=OUTPUT,source_release=SOURCE_RELEASE,
        source_row='BASE_CE-s2026092701',model_seed=2026092701,states=list(STATES),modes=list(MODES),
        LR=LR,optimizer_state='INITIAL empty originalmoments; FINAL20 savedm/v/step; both fixedstageLR2e-4',
        current_pairs=[[2,3],[19,20],[37,38],[54,55]],held_ordinals=[66,67],source_role='L_s',
        source_subset='reuse ROWconfirmation/meantransfer900; newmechanism sourceprobe, not independentnewdata confirmation',
        method=METHOD,source_falsification=GATE,original_default_head_joint=True,
        feature_scope='actual encoder F71 including3gains; W excluded from auxiliary',
        readouts=['ACTUAL','PRE_W'],PRE_W_gains='after-step actual gains retained',main_loss='original clean CE',
        head_fit_updates=0,U_IQ_read=False,V_IQ_read=False,U_truth_read=False,target_access=False,
        analysis_code_commit='release_commit.txt')

def validate(c):
    expected=request();expected['analysis_code_commit']=c.get('analysis_code_commit')
    if c!=expected or not isinstance(c['analysis_code_commit'],str) or len(c['analysis_code_commit'])!=40:
        raise ValueError('Fixed contrast-transfer request or Gitcommit differs')
    return c

def source_gate(records,paired):
    if [(r['state'],r['ordinal'],r['mode']) for r in records]!=[(s,o,m) for s in STATES for o in ORDINALS for m in MODES]:raise ValueError('All24 fixed contrast records required')
    if [(p['state'],p['ordinal']) for p in paired]!=[(s,o) for s in STATES for o in ORDINALS]:raise ValueError('All8 fixed comparisons required')
    qualified={s:[] for s in STATES};details=[]
    for p in paired:
        candidate=next(r for r in records if (r['state'],r['ordinal'],r['mode'])==(p['state'],p['ordinal'],'CROSS_RX'));reasons=[];values=[candidate['auxiliary_gradient_norm']]
        current=candidate['current_transfer']['CROSS_RX']
        if candidate['auxiliary_gradient_norm']<=0 or current['active_decisions']<GATE['current_active_decisions_min'] or current['active_origin_packet_count']<GATE['current_active_origin_packets_min']:reasons.append('insufficient_current_distinct_identity_opportunity')
        for control in GATE['controls']:
            q=p['vs_'+control]
            for key,minimum in [('actual_F_difference_norm',GATE['actual_F_difference_norm_min']),('held_feature_RMS_difference',GATE['held_feature_RMS_difference_min'])]:
                values.append(q[key])
                if q[key]<=minimum:reasons.append(control+':'+key)
            if q['shallow_middle_difference_tensors']<GATE['shallow_middle_difference_tensors_min']:reasons.append(control+':no_shallow_middle_change')
            values.append(q['held_transfer_CE_difference'])
            if q['held_transfer_CE_difference']>GATE['held_transfer_CE_difference_max']:reasons.append(control+':no_functional_transfer_CE_gain')
            if q['held_transfer_errors_increase']>GATE['held_transfer_errors_increase_tolerance']:reasons.append(control+':transfer_error_harm')
            for readout in GATE['readouts']:
                values.extend([q[readout+'_CE_relative_difference'],q[readout+'_accuracy_difference']])
                if q[readout+'_CE_relative_difference']>GATE['held_original_CE_relative_increase_tolerance']:reasons.append(control+':'+readout+'_relative_CE_harm')
                if q[readout+'_accuracy_difference']<GATE['held_original_accuracy_difference_min']:reasons.append(control+':'+readout+'_accuracy_harm')
        if not all(math.isfinite(float(v)) for v in values):raise ValueError('Nonfinite source contrast qualification')
        if not reasons:qualified[p['state']].append(p['ordinal'])
        details.append(dict(state=p['state'],ordinal=p['ordinal'],qualified=not reasons,failed_conditions=reasons))
    stage='SCRATCH20_SIGNAL' if len(qualified['INITIAL'])>=2 else 'E20_PLUS20_SIGNAL' if len(qualified['FINAL20'])>=2 else None
    return dict(passed=stage is not None,stage_signal=stage,qualified_ordinals_by_state=qualified,
        cell_conditions=details,controls_not_fallback=True,interpretation=GATE['interpretation'])
