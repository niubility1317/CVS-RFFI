"""Fixed source prototype for identity correspondence, not target prediction."""
from pathlib import Path
import math
from experiments.cvs_crossrx_sharpness20 import design as parent

ROOT=Path(__file__).resolve().parents[2]
PROJECT,read,write=parent.PROJECT,parent.read,parent.write
RUN='20261015-phase1-SIGTIME-identity-transport-sourceL-s2026092701-r01'
RELEASE='cvs_SIGTIME_identity_transport_sourceL_20261015_r01'
SOURCE_RELEASE=PROJECT+'/releases/'+parent.RELEASE
OUTPUT=PROJECT+'/runs/'+RUN+'/source_diagnostic'
MODES=('ZERO','SAME_BI','CROSS_ROW','CROSS_BI');STATES=('INITIAL','FINAL20');ORDINALS=(0,17,35,52)
SOURCE_RX=(1,3,4,6,8);LR=2e-4
METHOD=dict(cost='squared rawjoint distance/160, double center cost, divide by differentiable centered-cost RMS',
    RMS_floor=1e-8,temperature=.1,stop_probability=.9,weight=.1,Sinkhorn_iterations=80,
    BI_probability='conditional row and column quality from finite log-plan; exact-uniform limit equals6P',
    gradients='both actualjoint endpoints, W absent, all three gains included')
GATE=dict(minimum_qualified_cohorts_per_state=2,actual_F_difference_norm_min=1e-6,
    held_feature_RMS_difference_min=1e-5,shallow_middle_difference_tensors_min=1,
    held_correct_matching_mass_gain_vs_each_control_min=1e-4,
    held_matching_errors_increase_tolerance=0,held_CE_increase_tolerance=dict(INITIAL=1e-4,FINAL20=1e-6),
    controls=['ZERO','SAME_BI','CROSS_ROW'],readouts=['ACTUAL','PRE_W'],
    raw_cosine_margin='reported, not a qualification proxy: matching permits common receiver offsets',
    interpretation='True-label source identity correspondence signal at specified stage; not unseenRX/DG/novelty proof')

def request():
    return dict(run_id=RUN,output_root=OUTPUT,source_release=SOURCE_RELEASE,
        source_row='BASE_CE-s2026092701',model_seed=2026092701,states=list(STATES),modes=list(MODES),
        LR=LR,optimizer_state='INITIAL fresh original options; FINAL20 saved m/v/step; both fixedstageLR2e-4',
        current_pairs=[[0,1],[17,18],[35,36],[52,53]],held_ordinals=[68,69],source_role='L_s',
        method=METHOD,source_falsification=GATE,original_default_head_joint=True,
        feature_scope='actual encoder F71 including3gains; W excluded from auxiliary',
        readouts=['ACTUAL','PRE_W'],PRE_W_gains='after-step actual gains retained',
        main_loss='original clean CE',head_fit_updates=0,U_IQ_read=False,V_IQ_read=False,
        U_truth_read=False,target_access=False,analysis_code_commit='release_commit.txt')

def validate(c):
    expected=request();expected['analysis_code_commit']=c.get('analysis_code_commit')
    if c!=expected or not isinstance(c['analysis_code_commit'],str) or len(c['analysis_code_commit'])!=40:
        raise ValueError('Fixed identity matching request or actual Git commit differs')
    return c

def source_gate(records,paired):
    if [(r['state'],r['ordinal'],r['mode']) for r in records]!=[(s,o,m) for s in STATES for o in ORDINALS for m in MODES]:
        raise ValueError('All32 fixed state/cohort/method records required')
    if [(p['state'],p['ordinal']) for p in paired]!=[(s,o) for s in STATES for o in ORDINALS]:
        raise ValueError('All8 fixed paired cohorts required')
    qualified={s:[] for s in STATES};details=[]
    for p in paired:
        state,ordinal=p['state'],p['ordinal'];candidate=next(r for r in records if (r['state'],r['ordinal'],r['mode'])==(state,ordinal,'CROSS_BI'))
        reasons=[];values=[]
        if candidate['auxiliary_gradient_norm']<=0 or candidate['current_matching']['CROSS_BI']['active_rows_and_columns']==0:
            reasons.append('no_current_identity_matching_gradient')
        for control in GATE['controls']:
            q=p['vs_'+control]
            for key,minimum in [('actual_F_difference_norm',GATE['actual_F_difference_norm_min']),
                ('held_feature_RMS_difference',GATE['held_feature_RMS_difference_min']),
                ('held_correct_matching_mass_gain',GATE['held_correct_matching_mass_gain_vs_each_control_min'])]:
                values.append(q[key])
                if q[key]<=minimum:reasons.append(control+':'+key)
            if q['shallow_middle_difference_tensors']<GATE['shallow_middle_difference_tensors_min']:
                reasons.append(control+':no_shallow_middle_change')
            if q['held_matching_errors_increase']>GATE['held_matching_errors_increase_tolerance']:
                reasons.append(control+':matching_error_harm')
            for readout in GATE['readouts']:
                value=q[readout+'_CE_difference'];values.append(value)
                if value>GATE['held_CE_increase_tolerance'][state]:reasons.append(control+':'+readout+'_CE_harm')
        if not all(math.isfinite(float(v)) for v in values):raise ValueError('Nonfinite source qualification')
        if not reasons:qualified[state].append(ordinal)
        details.append(dict(state=state,ordinal=ordinal,qualified=not reasons,failed_conditions=reasons))
    stage='SCRATCH20_SIGNAL' if len(qualified['INITIAL'])>=2 else 'E20_PLUS20_SIGNAL' if len(qualified['FINAL20'])>=2 else None
    return dict(passed=stage is not None,stage_signal=stage,qualified_ordinals_by_state=qualified,
        cell_conditions=details,controls_not_fallback=True,interpretation=GATE['interpretation'])
