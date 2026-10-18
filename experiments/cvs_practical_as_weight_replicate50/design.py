"""Conditional fixed02/03 original20+weighted10+weighted20; preparation only until sourcePASS."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_practical_as_weight50 import design as weighted50
from experiments.cvs_practical_as_weight30 import design as weighted30
ROOT=Path(__file__).resolve().parents[2]
RUN='20261018-phase1-FcrossRX-Asweight-e50-s0203-m4-r01';RELEASE='cvs_FcrossRX_AsWeight_E50_s0203_20261018_r01'
OWNER='codex/root/practical-Asweight-replicate50';LOCAL='local_artifacts/F_As_weight_E50_s0203_20261018'
PARENT_RUN='20261018-phase1-FcrossRX-As-replication-e30-s0203-m4-r02';PARENT_RELEASE='cvs_FcrossRX_As_replication_E30_s0203_20261018_r02';PARENT_COMMIT='63b7c8cd7bb4139a2d800e6f4c4a7f1f2163fd47'
WEIGHTED50_COMMIT='64176e7ff1aa5cd2477a504f8b442bb7966079dc';WEIGHTED30_COMMIT='3befd47c8041d81f9430eda53562d0821624ff04'
REFERENCE_COMMIT='19aedc3604fed9c193237dd92366164178af0983';REFERENCE_RELEASE=weighted50.REFERENCE_RELEASE;REFERENCE_RUN=weighted50.REFERENCE_RUN
SEED01_WEIGHT50_RUN=weighted50.RUN
SEEDS=(2026092702,2026092703);ARMS=('B00','B01');RX=FIT=weighted50.RX;STEPS=222
PROJECT=weighted50.PROJECT;SOURCE=weighted50.SOURCE;FORMAL_VIEWS=weighted50.FORMAL_VIEWS;TRUTH=weighted50.TRUTH;PANEL=weighted50.PANEL;SOURCE_CACHE=weighted50.SOURCE_CACHE;FULL_FP32_POLICY=weighted50.FULL_FP32_POLICY;CLASSES=weighted50.CLASSES;SCENES=weighted50.SCENES;VIEWS=weighted50.VIEWS
BASE=Path(PROJECT)/'runs'/RUN;read,write=weighted50.read,weighted50.write
ELIGIBILITY=Path(PROJECT)/'runs'/SEED01_WEIGHT50_RUN/'analysis_auto_r01/source_summary.json'

def rows():return [dict(arm=a,row_id=a+'-s'+str(s),model_seed=s) for s in SEEDS for a in ARMS]
def stage_root(row,budget):
 if row not in rows() or budget not in (30,50):raise ValueError('Only four fixed trajectories/two budgets')
 return BASE/row['row_id']/('weighted'+str(budget))

def reference_config(c,segment=30):
 from experiments.cvs_As_replicate49 import design as original
 return original.config(next(r for r in original.rows() if r['row_id']==c['row_id']),segment)

def config(row,budget=50):
 if row not in rows() or budget not in (30,50):raise ValueError('Fixed02/03 and ownweighted30/50 only')
 template=weighted30 if budget==30 else weighted50;c=deepcopy(template.config(next(r for r in template.rows() if r['arm']==row['arm'])));parent=Path(PROJECT)/'runs'/PARENT_RUN/row['row_id']/'stage20' if budget==30 else stage_root(row,30)/'source'
 c.update(schema='FcrossRX_Asweight_seed_replication50_v1',run_id=RUN,**row,target_budget=budget,launch_owner=OWNER,output_root=str(stage_root(row,budget)/'source'),parent_source=str(parent),checkpoint_sources=[str(parent/'final_ssdg.pth')],model_initialization='strict own original20' if budget==30 else 'strict own newweighted30 student and EMA',source_selection='fixedFINAL30 then unconditional fixedFINAL50;intermediate source/formal values never control continuation',budget_authorization='conditional fixedsource01 eligibility before release;four trajectories each original20+new10+new20;120newmodelEpoch/26640updates;no seed01 rerun',baseline_reference='same fixed seed01 algorithm;02/03 true factory/native seeds and own original20;24logical original/weighted x NoAs/As x30/50 x3seeds',benchmark_context='No registration or release until fixed seed01 source50 conjunction passes;once released all4x2stages unconditional scientifically,technical failures preserved')
 c['seed_binding']=dict(model_and_native_training=row['model_seed'],F_factory_model_seed=row['model_seed'],physical_split=392005,physical_roles='unchanged exact IDs',train_channel=20261016421,validation_channel=20261016422,receiver=20261016423,legacy_reference_config='recipe only;not seed or initializer')
 return c

def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row,c.get('target_budget')):raise ValueError('Fixed02/03 staged data/factory/parent/budget differs')
 return c

def make_args(c,device='cuda:0'):
 validate(c)
 from experiments.cvs_As_replicate49 import design as original
 a=original.make_args(reference_config(c,30),device);a.seed=c['model_seed'];a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];a.baseline_ckpt=c['parent_source']+'/final_ssdg.pth';a.teacher_ckpt='';a.epochs=c['epochs'];a.label_epochs=0;a.pseudo_epochs=c['epochs'];a.lr=.0002;a.weight_decay=.0001;a.from_scratch=False;a.a1_scratch_only=False;a.a1_ema_startup_average=False
 if a.seed!=c['model_seed'] or a.batch_size!=128 or a.label_smoothing!=.01 or not a.use_aug or a.use_concat_sat_channel_aug or not a.use_ema_teacher:raise ValueError('Actual seed/L/U/native defaulthead recipe differs')
 return a
