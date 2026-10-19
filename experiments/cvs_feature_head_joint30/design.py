from copy import deepcopy
from pathlib import Path
from experiments.cvs_feature_adapters30 import design as frozen
read,write=frozen.read,frozen.write
ROOT=Path(__file__).resolve().parents[2]
RUN='20261019-phase1-F-headJoint-e20plus10-s2026092701-m6-r01';RELEASE='cvs_F_featureHeadJoint_E30_20261019_r01';LOCAL='local_artifacts/F_featureHeadJoint_E30_20261019';OWNER='codex/root/F-feature-head-joint'
PROJECT=read(ROOT/'runtime.json')['project'] if (ROOT/'runtime.json').is_file() else 'PROJECT';BASE=Path(PROJECT)/'runs'/RUN
SEED,RANK,KERNEL,ALPHA,EPOCHS,STEPS,UPDATES,BATCH=frozen.SEED,frozen.RANK,frozen.KERNEL,frozen.ALPHA,10,222,2220,128
SOURCE,FORMAL_VIEWS,TRUTH,PANEL=frozen.SOURCE,frozen.FORMAL_VIEWS,frozen.TRUTH,frozen.PANEL
FIT,SCENES,VIEWS,CLASSES,FULL_FP32_POLICY=frozen.FIT,frozen.SCENES,frozen.VIEWS,frozen.CLASSES,frozen.FULL_FP32_POLICY
REFERENCE_RUN,PARENT_ROW,HEAD=frozen.REFERENCE_RUN,frozen.PARENT_ROW,frozen.HEAD
FROZEN_RUN=frozen.RUN;FROZEN_BASE=Path(PROJECT)/'runs'/FROZEN_RUN
REFERENCE_RELEASE=frozen.RELEASE;REFERENCE_COMMIT='d53c53982d3bd98dc817a41b90f4adab0556b099'
parent_config=frozen.parent_config

def rows():return [dict(r,row_id=r['arm']+'_JOINT-s'+str(SEED)) for r in frozen.rows()]

def frozen_config(row):return frozen.config(next(r for r in frozen.rows() if r['arm']==row['arm']))

def config(row):
 if row not in rows():raise ValueError('Unregistered default-head joint six-row matrix')
 c=deepcopy(frozen_config(row));c.update(schema='F_feature_head_joint30_v1',run_id=RUN,**row,launch_owner=OWNER,output_root=str(BASE/row['row_id']/'source'),fixed_head_control_run=FROZEN_RUN,observation_bank=str(FROZEN_BASE/'observation_bank'))
 c['training']['default_head_frozen']=False;c['training']['default_head_joint']=True;c['training']['loss']='CE0(raw)+CE0(second)+mean_all128[parentCorrect*KL(frozen_B0_parent_raw||joint_feature_and_original_W_student_raw)]'
 c['optimizer']['classifier']='existing W, same AdamW lr/eps/weight_decay and same global clip; no new/reset head or head-only optimizer'
 c['source_promotion']=dict(joint_minus_same_frozen_As_urban_min_pp=.5,joint_minus_same_frozen_As_fullVclean_min_pp=-.2,joint_As_minus_joint_NoAs_fullVclean_min_pp=-.2,adapter_minus_FULL_joint_As_urban_min_pp=.5,adapter_minus_FULL_frozen_As_urban_min_pp=.5,automatic_extension=False,max_cumulative_epochs=50,on_fail='close this fixed joint-W comparison;no W-lr/rank/kernel/lambda/seed rescue')
 return c

def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed head-joint/source/parent/stream/budget recipe differs')
 return c
