"""Fixed source-L identity-difference falsification before new training."""
from pathlib import Path
from experiments.cvs_crossrx_sharpness20 import design as parent

ROOT=Path(__file__).resolve().parents[2]
PROJECT,read,write=parent.PROJECT,parent.read,parent.write
RUN='20261015-phase1-SIGTIME-crossRX-identity-difference-sourceL-s2026092701-r01'
RELEASE='cvs_SIGTIME_crossRX_identity_difference_sourceL_20261015_r01'
SOURCE_RELEASE=PROJECT+'/releases/'+parent.RELEASE
OUTPUT=PROJECT+'/runs/'+RUN+'/source_diagnostic'
STATES=('INITIAL','FINAL20');ORDINALS=(0,17,35,52)
MODES=('ZERO','SELF_DIFF','CROSS_DIFF');SOURCE_RX=(1,3,4,6,8)
GATE=dict(minimum_FINAL20_qualified_cells=2,CROSS_active_direction_pairs_min=30,
    delta_norm_min=1e-4,CROSS_SELF_auxiliary_gradient_relative_difference_min=.01,
    CROSS_ZERO_actual_F_displacement_relative_difference_min=.01,
    CROSS_ZERO_held_feature_RMS_min=1e-7,shallow_middle_tensors_changed_min=1,
    CROSS_ZERO_held_identity_margin_gain_min=1e-5,CROSS_SELF_held_identity_margin_gain_min=1e-5,
    CROSS_ZERO_held_mean_CE_increase_max=1e-6,CROSS_SELF_held_mean_CE_increase_max=1e-6,
    interpretation='Known-source finite-step feature signal only; FINAL20 gate for further E20+20 investigation, no actual training/target/novelty proof')


def request():
    return dict(run_id=RUN,output_root=OUTPUT,source_release=SOURCE_RELEASE,
        source_row='BASE_CE-s2026092701',model_seed=2026092701,
        states=list(STATES),modes=list(MODES),anchor_ordinals=list(ORDINALS),
        current_pairs=[[0,1],[17,18],[35,36],[52,53]],held_ordinals=[68,69],
        source_role='L_s',V_IQ_read=False,U_IQ_read=False,U_truth_read=False,target_access=False,
        coefficient=.1,direction_stop=.9,normalization_eps=1e-4,
        source_falsification=GATE,analysis_code_commit='release_commit.txt')


def validate(c):
    expected=request();expected['analysis_code_commit']=c.get('analysis_code_commit')
    if c!=expected or not isinstance(c['analysis_code_commit'],str) or len(c['analysis_code_commit'])!=40:
        raise ValueError('Fixed identity-difference request or actual Git commit differs')
    return c


def source_gate(records):
    if [(r['state'],r['ordinal']) for r in records]!=[(s,o) for s in STATES for o in ORDINALS]:
        raise ValueError('All eight fixed source cells required')
    qualified=[]
    for row in records:
        if (row['CROSS_active_direction_pairs']>=GATE['CROSS_active_direction_pairs_min']
            and row['minimum_class_difference_norm']>=GATE['delta_norm_min']
            and row['CROSS_SELF_auxiliary_gradient_relative_difference']>GATE['CROSS_SELF_auxiliary_gradient_relative_difference_min']
            and row['CROSS_ZERO_actual_F_displacement_relative_difference']>GATE['CROSS_ZERO_actual_F_displacement_relative_difference_min']
            and row['CROSS_ZERO_held_feature_RMS']>GATE['CROSS_ZERO_held_feature_RMS_min']
            and row['CROSS_ZERO_shallow_middle_tensors_changed']>=GATE['shallow_middle_tensors_changed_min']
            and row['CROSS_ZERO_held_identity_margin_gain']>GATE['CROSS_ZERO_held_identity_margin_gain_min']
            and row['CROSS_SELF_held_identity_margin_gain']>GATE['CROSS_SELF_held_identity_margin_gain_min']
            and row['CROSS_ZERO_held_mean_CE_increase']<=GATE['CROSS_ZERO_held_mean_CE_increase_max']
            and row['CROSS_SELF_held_mean_CE_increase']<=GATE['CROSS_SELF_held_mean_CE_increase_max']):
            qualified.append(dict(state=row['state'],ordinal=row['ordinal']))
    counts={s:sum(row['state']==s for row in qualified) for s in STATES}
    return dict(passed=counts['FINAL20']>=GATE['minimum_FINAL20_qualified_cells'],
        qualified_cells=qualified,qualified_by_state=counts,
        INITIAL_not_a_substitute_for_FINAL20=True,interpretation=GATE['interpretation'])
