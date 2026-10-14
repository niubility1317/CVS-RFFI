"""Fixed pure-CE continuation-state diagnostic, without new loss."""
from pathlib import Path
from experiments.cvs_crossrx_sharpness20 import design as parent

ROOT=Path(__file__).resolve().parents[2]
PROJECT,read,write=parent.PROJECT,parent.read,parent.write
RUN='20261015-phase1-SIGTIME-pureCE-restart-state-sourceL-s2026092701-r01'
RELEASE='cvs_SIGTIME_pureCE_restart_state_sourceL_20261015_r01'
SOURCE_RELEASE=PROJECT+'/releases/'+parent.RELEASE
OUTPUT=PROJECT+'/runs/'+RUN+'/source_diagnostic'
REGIMES=('END_SAVED','RESTART_KEEP','RESTART_RESET');ORDINALS=(0,17,35,52)
SOURCE_RX=(1,3,4,6,8);RESTART_LR=2e-4
GATE=dict(minimum_RESET_qualified_cells=2,RESET_actual_F_step_norm_min=1e-6,
    RESET_held_feature_RMS_min=1e-5,RESET_shallow_middle_tensors_changed_min=1,
    RESET_preW_held_CE_reduction_min=1e-6,RESET_held_raw_identity_margin_gain_min=1e-5,
    interpretation='Intended reset-stage pureCE source feature signal only; neither DG efficacy nor revival of a closed auxiliary candidate')


def request():
    return dict(run_id=RUN,output_root=OUTPUT,source_release=SOURCE_RELEASE,
        source_row='BASE_CE-s2026092701',model_seed=2026092701,state='FINAL20',
        regimes=list(REGIMES),restart_lr=RESTART_LR,current_pairs=[[0,1],[17,18],[35,36],[52,53]],
        held_ordinals=[68,69],source_role='L_s',loss='original clean CE',
        original_default_head_joint=True,new_loss=False,head_fit_updates=0,
        U_IQ_read=False,V_IQ_read=False,U_truth_read=False,target_access=False,
        feature_scope='actual encoder including3gains; W excluded from f_theta',
        readouts=['ACTUAL','PRE_W'],PRE_W_gains='after-step actual gains retained',
        source_falsification=GATE,analysis_code_commit='release_commit.txt')


def validate(c):
    expected=request();expected['analysis_code_commit']=c.get('analysis_code_commit')
    if c!=expected or not isinstance(c['analysis_code_commit'],str) or len(c['analysis_code_commit'])!=40:
        raise ValueError('Fixed pureCE stage request or actual Git commit differs')
    return c


def source_gate(records):
    if [(r['ordinal'],r['regime']) for r in records]!=[(o,m) for o in ORDINALS for m in REGIMES]:
        raise ValueError('All twelve fixed regime/cohort records required')
    qualified=[]
    for row in records:
        if (row['regime']=='RESTART_RESET'
            and row['actual_F_step_norm']>GATE['RESET_actual_F_step_norm_min']
            and row['held_feature_RMS_from_before']>GATE['RESET_held_feature_RMS_min']
            and row['shallow_middle_tensors_changed_from_before']>=GATE['RESET_shallow_middle_tensors_changed_min']
            and row['PRE_W_held_CE_reduction_from_before']>GATE['RESET_preW_held_CE_reduction_min']
            and row['held_raw_identity_margin_gain_from_before']>GATE['RESET_held_raw_identity_margin_gain_min']):
            qualified.append(row['ordinal'])
    return dict(passed=len(qualified)>=GATE['minimum_RESET_qualified_cells'],RESET_qualified_ordinals=qualified,
        KEEP_or_END_not_a_substitute_for_RESET=True,interpretation=GATE['interpretation'])
