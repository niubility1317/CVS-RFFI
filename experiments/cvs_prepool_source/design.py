from pathlib import Path
from experiments.cvs_feature_head_joint30 import design as joint
read,write=joint.read,joint.write
ROOT=Path(__file__).resolve().parents[2]
RUN='20261019-phase1-F-prepool-source-s2026092701-r01';RELEASE='cvs_F_prepool_source_20261019_r01';LOCAL='local_artifacts/F_prepool_source_20261019';OWNER='codex/root/F-prepool-source'
PROJECT=read(ROOT/'runtime.json')['project'] if (ROOT/'runtime.json').is_file() else 'PROJECT';BASE=Path(PROJECT)/'runs'/RUN
REFERENCE_RELEASE=joint.RELEASE;REFERENCE_COMMIT='931af701470e781122494398dc58c3e027597550'
SEED=2026092701;DIAGNOSTIC_SEED=20261019821;ROW='prepool-source'
def config():return dict(schema='F_prepool_source_v1',run_id=RUN,row_id=ROW,launch_owner=OWNER,model_seed=SEED,diagnostic_seed=DIAGNOSTIC_SEED,models=['B0_FINAL20','FULL_As_JOINT_FINAL30'],source_bank_run=joint.FROZEN_RUN,L_physical=1800,source_RX=[1,3,4,6,8],days=[1,2,3],TX=6,views=['raw','mid_draw0','low_draw0','mid_draw1','low_draw1'],NN=18000,CPU_threads=2,device='cpu',fit_updates=0,U_used=False,V_used=False,target_access=False,new_physical_simulations=0,descriptor='mean(v),mean(v^2),mean(abs(v)) over time;v=pair mean or pair half-difference',statistic='within RX/day centered class-mean cross-half inner product S divided by total descriptor scatter T;sumS/sumT;no nonnegative clipping;no classifier fit',selection='fixed plan;no target feedback;no seed/draw/descriptor sweep')
