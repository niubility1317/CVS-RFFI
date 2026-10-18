"""Fixed seed02/03 replication of original As/NoAs own-FINAL30+20 E50 recipe."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_practical_budget50 import design as old
ROOT=Path(__file__).resolve().parents[2]
RUN='20261018-phase1-FcrossRX-As-budget-e50-s0203-m4-r01'
RELEASE='cvs_FcrossRX_As_budget_E50_s0203_20261018_r01'
OWNER='codex/root/practical-budget50-replication';LOCAL='local_artifacts/F_As_budget_E50_s0203_20261018'
REFERENCE_RUN='20261018-phase1-FcrossRX-As-replication-e30-s0203-m4-r02'
REFERENCE_COMMIT='63b7c8cd7bb4139a2d800e6f4c4a7f1f2163fd47'
P43_COMMIT='19aedc3604fed9c193237dd92366164178af0983'
SEED01_RUN=old.RUN
SEEDS=(2026092702,2026092703);ARMS=('B00','B01');RX=FIT=old.RX;STEPS=old.STEPS;UPDATES=old.UPDATES
PROJECT=old.PROJECT;SOURCE=old.SOURCE;FULL_FP32_POLICY=old.FULL_FP32_POLICY;CLASSES=old.CLASSES;SCENES=old.SCENES;VIEWS=old.VIEWS;FORMAL_VIEWS=old.FORMAL_VIEWS;TRUTH=old.TRUTH;PANEL=old.PANEL;SOURCE_CACHE=old.SOURCE_CACHE
BASE=Path(PROJECT)/'runs'/RUN
PARENT_EPOCHS=30;EPOCH_OFFSET=30;STEP_OFFSET=6660;CUMULATIVE_EPOCHS=50;CUMULATIVE_UPDATES=11100
read,write=old.read,old.write

def rows():return [dict(arm=a,row_id=a+'-s'+str(s),model_seed=s) for s in SEEDS for a in ARMS]
def reference_config(c,segment=30):
 from experiments.cvs_As_replicate49 import design as b
 row=next(r for r in b.rows() if r['row_id']==c['row_id'])
 return b.config(row,segment)
def config(row,segment=30):
 if row not in rows() or segment!=30:raise ValueError('Only fixed02/03 originalNoAs/As ownFINAL30+20')
 c=deepcopy(old.config(next(r for r in old.rows() if r['arm']==row['arm']),30));parent=Path(PROJECT)/'runs'/REFERENCE_RUN/row['row_id']/'source'
 c.update(schema='FcrossRX_practical_budget50_replication_v1',run_id=RUN,**row,launch_owner=OWNER,output_root=str(BASE/row['row_id']/'source'),parent_source=str(parent),checkpoint_sources=[str(parent/'final_ssdg.pth')],budget_authorization='Fixed original seed02/03 replication of E50 from three-seed legal source budget evidence;ownFINAL30+20 each;no seed01 retraining',cost_claim='Pernewrow20epochs/4440updates/6271922nativeNN+12600source+1176002formal=7460524NN;4newrows;historical30 and seed01 E50 separate',source_selection='FixedFINAL50/all four rows;source readout never chooses epoch/seed or skips formal;no automatic expansion',benchmark_context='Prior exposedbenchmark context retained;fixed pre-existing seed02/03 order and unchanged E50 recipe;formal scores never select seeds or parameters')
 c['seed_binding']=dict(model_and_native_training=row['model_seed'],physical_split=392005,physical_roles='unchanged exact IDs',F_factory_model_seed=row['model_seed'],train_channel=20261016421,validation_channel=20261016422,receiver=20261016423,legacy_reference_config='architecture/recipe reference only;never checkpoint or actual model seed')
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row,c.get('segment')):raise ValueError('Fixed E50 replication data/seed/parent/budget/head/physics differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c)
 from experiments.cvs_As_replicate49 import design as b
 a=b.make_args(reference_config(c,30),device)
 a.seed=c['model_seed'];a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];a.baseline_ckpt=c['parent_source']+'/final_ssdg.pth';a.teacher_ckpt='';a.epochs=20;a.label_epochs=0;a.pseudo_epochs=20;a.lr=.0002;a.weight_decay=.0001;a.from_scratch=False;a.a1_scratch_only=False;a.a1_ema_startup_average=False
 if a.seed!=c['model_seed'] or a.label_smoothing!=.01 or not a.use_aug or a.use_concat_sat_channel_aug or not a.use_ema_teacher or a.batch_size!=128:raise ValueError('Original seed/L/RDrop/U arguments differ')
 return a
