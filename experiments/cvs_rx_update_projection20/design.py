"""A fixed source-L falsification before new actual training."""
from pathlib import Path
from experiments.cvs_crossrx_sharpness20 import design as parent

ROOT=Path(__file__).resolve().parents[2]
PROJECT,read,write=parent.PROJECT,parent.read,parent.write
RUN='20261015-phase1-SIGTIME-RX-update-projection-sourceL-s2026092701-r01'
RELEASE='cvs_SIGTIME_RX_update_projection_sourceL_20261015_r01'
SOURCE_RELEASE=PROJECT+'/releases/'+parent.RELEASE
OUTPUT=PROJECT+'/runs/'+RUN+'/source_diagnostic'
STATES=('INITIAL','FINAL20');ORDINALS=(0,17,35,52)
RISK_ORDINALS=(66,67);HELD_ORDINALS=(68,69)
MODES=('RAW','GRAD_CONE','UPDATE_CONE')
SOURCE_RX=(1,3,4,6,8)
GATE=dict(minimum_qualified_cells=2,minimum_RAW_positive_risk_RX=1,
    UPDATE_RAW_relative_displacement_difference_min=.01,
    UPDATE_retained_actual_displacement_ratio_min=.5,
    UPDATE_RAW_held_worst_RX_CE_delta_reduction_min=1e-6,
    UPDATE_RAW_held_mean_CE_increase_max=1e-6,
    UPDATE_RAW_held_feature_RMS_min=1e-7,
    minimum_shallow_or_middle_tensors_changed_from_RAW=1,
    minimum_UPDATE_GRAD_distinct_cells=2,UPDATE_GRAD_relative_displacement_difference_min=.01,
    interpretation='Known-source actual displacement and independent-L finite-step signal only; no training efficacy, target selection or novelty proof')


def request():
    return dict(run_id=RUN,output_root=OUTPUT,source_release=SOURCE_RELEASE,
        source_row='BASE_CE-s2026092701',model_seed=2026092701,
        states=list(STATES),modes=list(MODES),anchor_ordinals=list(ORDINALS),
        risk_ordinals=list(RISK_ORDINALS),held_ordinals=list(HELD_ORDINALS),
        source_role='L_s',U_truth_read=False,V_gradient=False,target_access=False,
        source_falsification=GATE,analysis_code_commit='release_commit.txt')


def validate(c):
    expected=request();expected['analysis_code_commit']=c.get('analysis_code_commit')
    if c!=expected or not isinstance(c['analysis_code_commit'],str) or len(c['analysis_code_commit'])!=40:
        raise ValueError('Fixed source update projection request or actual Git commit differs')
    return c


def source_gate(records):
    if [(r['state'],r['ordinal']) for r in records]!=[(s,o) for s in STATES for o in ORDINALS]:
        raise ValueError('All eight preregistered source cells are required')
    qualified=[];distinct=[]
    for r in records:
        update=r['modes']['UPDATE_CONE'];q=update['actual_displacement']
        if r['UPDATE_GRAD_relative_displacement_difference']>GATE['UPDATE_GRAD_relative_displacement_difference_min']:
            distinct.append(dict(state=r['state'],ordinal=r['ordinal']))
        if (r['RAW_positive_directional_risk_RX']>=GATE['minimum_RAW_positive_risk_RX']
            and r['UPDATE_RAW_relative_displacement_difference']>GATE['UPDATE_RAW_relative_displacement_difference_min']
            and r['UPDATE_retained_actual_displacement_ratio']>=GATE['UPDATE_retained_actual_displacement_ratio_min']
            and r['UPDATE_RAW_held_worst_RX_CE_delta_reduction']>GATE['UPDATE_RAW_held_worst_RX_CE_delta_reduction_min']
            and r['UPDATE_RAW_held_mean_CE_increase']<=GATE['UPDATE_RAW_held_mean_CE_increase_max']
            and r['UPDATE_RAW_held_feature_RMS']>GATE['UPDATE_RAW_held_feature_RMS_min']
            and r['shallow_middle_tensors_changed_from_RAW']>=GATE['minimum_shallow_or_middle_tensors_changed_from_RAW']
            and r['actual_risk_within_solver_and_rounding_bound'] and q['actual_parameter_addition_measured']):
            qualified.append(dict(state=r['state'],ordinal=r['ordinal']))
    return dict(passed=len(qualified)>=GATE['minimum_qualified_cells'] and len(distinct)>=GATE['minimum_UPDATE_GRAD_distinct_cells'],
        qualified_cells=qualified,UPDATE_GRAD_distinct_cells=distinct,
        qualified_by_state={s:sum(q['state']==s for q in qualified) for s in STATES})
