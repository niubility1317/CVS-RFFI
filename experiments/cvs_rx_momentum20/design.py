"""Fixed source-only falsification, not a training or target-selected recipe."""
from pathlib import Path
import math
from experiments.cvs_crossrx_sharpness20 import design as parent

ROOT=Path(__file__).resolve().parents[2]
PROJECT,read,write=parent.PROJECT,parent.read,parent.write
RUN='20261015-phase1-SIGTIME-RX-history-momentum-sourceL-s2026092701-r01'
RELEASE='cvs_SIGTIME_RX_history_momentum_sourceL_20261015_r01'
SOURCE_RELEASE=PROJECT+'/releases/'+parent.RELEASE
OUTPUT=PROJECT+'/runs/'+RUN+'/source_diagnostic'
REGIMES=('KEEP','M_ZERO_F','RX_M_FILTER');ORDINALS=(0,17,35,52)
SOURCE_RX=(1,3,4,6,8);RESTART_LR=2e-4
RANK_RULE=dict(relative_eigenvalue_cutoff=1e-10,absolute_eigenvalue_cutoff=1e-30,maximum_rank=4)
GATE=dict(minimum_qualified_cells=2,actual_F_difference_norm_min=1e-6,
    held_feature_RMS_difference_min=1e-5,shallow_middle_difference_tensors_min=1,
    raw_identity_margin_gain_from_before_min=0.0,raw_identity_margin_gain_vs_each_control_min=1e-5,
    held_CE_increase_tolerance=1e-6,readouts=['PRE_W','ACTUAL'],
    interpretation='New historical-momentum filtering source signal against BOTH KEEP and M_ZERO_F; not DG efficacy or a main-training authorization by target score')


def request():
    return dict(run_id=RUN,output_root=OUTPUT,source_release=SOURCE_RELEASE,
        source_row='BASE_CE-s2026092701',model_seed=2026092701,state='FINAL20',
        regimes=list(REGIMES),restart_lr=RESTART_LR,current_pairs=[[0,1],[17,18],[35,36],[52,53]],
        held_ordinals=[68,69],source_role='L_s',loss='original clean CE',
        source_rx=list(SOURCE_RX),RX_gradient_packets=36,rank_rule=RANK_RULE,
        original_default_head_joint=True,current_gradient_preserved=True,second_moment_and_step_preserved=True,
        feature_scope='actual encoder including3gains; W excluded from history filtering',
        readouts=['ACTUAL','PRE_W'],PRE_W_gains='after-step actual gains retained',
        head_fit_updates=0,U_IQ_read=False,V_IQ_read=False,U_truth_read=False,target_access=False,
        source_falsification=GATE,analysis_code_commit='release_commit.txt')


def validate(c):
    expected=request();expected['analysis_code_commit']=c.get('analysis_code_commit')
    if c!=expected or not isinstance(c['analysis_code_commit'],str) or len(c['analysis_code_commit'])!=40:
        raise ValueError('Fixed RX history request or actual Git commit differs')
    return c


def source_gate(records,paired):
    if [(r['ordinal'],r['regime']) for r in records]!=[(o,m) for o in ORDINALS for m in REGIMES]:
        raise ValueError('All twelve fixed regime/cohort records required')
    if [p['ordinal'] for p in paired]!=list(ORDINALS):raise ValueError('Four paired cohorts required')
    qualified=[];details=[]
    for ordinal,pair in zip(ORDINALS,paired):
        by={r['regime']:r for r in records if r['ordinal']==ordinal};candidate=by['RX_M_FILTER']
        values=[candidate['held_raw_identity_margin_gain_from_before']]
        reasons=[]
        if not candidate['momentum_filter']['rank']:reasons.append('zero_contrast_rank')
        if candidate['held_raw_identity_margin_gain_from_before']<=GATE['raw_identity_margin_gain_from_before_min']:
            reasons.append('no_raw_identity_margin_improvement_from_before')
        for control in REGIMES[:2]:
            q=pair['vs_'+control]
            required=dict(actual_F_difference_norm=GATE['actual_F_difference_norm_min'],
                held_feature_RMS_difference=GATE['held_feature_RMS_difference_min'],
                raw_identity_margin_gain=GATE['raw_identity_margin_gain_vs_each_control_min'])
            for key,minimum in required.items():
                values.append(q[key])
                if q[key]<=minimum:reasons.append(control+':'+key)
            if q['shallow_middle_difference_tensors']<GATE['shallow_middle_difference_tensors_min']:
                reasons.append(control+':no_shallow_middle_change')
            for readout in GATE['readouts']:
                delta=q[readout+'_CE_difference'];values.append(delta)
                if delta>GATE['held_CE_increase_tolerance']:reasons.append(control+':'+readout+'_CE_harm')
        for readout in GATE['readouts']:
            reduction=candidate[readout+'_held_CE_reduction_from_before'];values.append(reduction)
            if reduction < -GATE['held_CE_increase_tolerance']:reasons.append(readout+':CE_harm_from_before')
        if not all(math.isfinite(float(value)) for value in values):raise ValueError('Nonfinite source qualification')
        if not reasons:qualified.append(ordinal)
        details.append(dict(ordinal=ordinal,qualified=not reasons,failed_conditions=reasons))
    return dict(passed=len(qualified)>=GATE['minimum_qualified_cells'],qualified_ordinals=qualified,
        cell_conditions=details,controls_are_not_fallback_candidates=True,interpretation=GATE['interpretation'])
