from pathlib import Path
from experiments.cvs_feature_head_joint30 import design as joint
read,write=joint.read,joint.write
ROOT=Path(__file__).resolve().parents[2]
RUN='20261019-phase1-F-sigBoundary-source-s2026092701-r01';RELEASE='cvs_F_sigBoundary_source_20261019_r01';LOCAL='local_artifacts/F_sigBoundary_source_20261019';OWNER='codex/root/F-sig-boundary-source'
PROJECT=read(ROOT/'runtime.json')['project'] if (ROOT/'runtime.json').is_file() else 'PROJECT';BASE=Path(PROJECT)/'runs'/RUN
REFERENCE_RELEASE=joint.RELEASE;REFERENCE_COMMIT='931af701470e781122494398dc58c3e027597550'
ROW='sig-boundary-source';MODES=('NATIVE','E','I','EI');MODELS=('B0_FINAL20','FULL_As_JOINT_FINAL30');VIEWS=('raw','actual_first','mid_draw0','low_draw0','mid_draw1','low_draw1')
def config():return dict(schema='F_sig_boundary_source_v1',run_id=RUN,row_id=ROW,launch_owner=OWNER,models=list(MODELS),model_seed=2026092701,source_bank_run=joint.FROZEN_RUN,L_physical=1800,L_full=6300,source_RX=[1,3,4,6,8],days=[1,2,3],TX=6,views=list(VIEWS),modes=list(MODES),operation='per8x6 native path window:mean-repeat quarters0/3 for E,1/2 for I;EI both;all3views',whole_encoder_NN=21600,real_CP_zeroIQ_smoke_NN=4,total_NN=21604,additional_alternative_head_decisions=64800,additional_native_replay_head_decisions=21600,additional_default_head_decisions=86400,total_default_head_decisions_source=108000,additional_pool_output_projections=86400,smoke_total_head_decisions=20,fit_updates=0,U_used=False,V_used=False,target_access=False,new_physical_simulations=0,CPU_threads=2,GPU_used=False,criteria=dict(model='FULL_As_JOINT_FINAL30',E_minus_NATIVE_fresh_midlow_min_pp=.5,E_minus_I_fresh_midlow_min_pp=.5,E_minus_NATIVE_raw_L_min_pp=-.2),automatic_training=False,automatic_extension=False,parameter_or_seed_sweep=False)
def validate(c):
 if c!=config():raise ValueError('Fixed source operator/model/role/budget differs')
 return c
