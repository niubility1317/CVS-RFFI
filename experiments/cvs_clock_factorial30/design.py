"""Fixed all-row E20+10 amendment; target results never choose a parent."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_clock_factorial20 import design as clock
from experiments.cvs_fullsource_concat20 import design as ref
ROOT=Path(__file__).resolve().parents[2]
RUN='20261017-phase1-F-crossRX-clock-factorial-e20plus10-s2026092701-m5-r02'
RELEASE='cvs_F_clock_factorial_e20plus10_20261017_r02'
LOCAL='local_artifacts/F_clock_factorial30_20261017_r02';OWNER='codex/root/F-clock-E30-20261017'
read,write=ref.read,ref.write
SOURCE,CAPSULE,TRUTH,VIEWS_ROOT=ref.SOURCE,ref.CAPSULE,ref.TRUTH,ref.VIEWS_ROOT
PROJECT,AF_RELEASE=ref.PROJECT,ref.AF_RELEASE
SEED,STEPS,UPDATES,FIT=ref.SEED,222,2220,ref.FIT
CLASSES,FULL_FP32_POLICY,SCENES,VIEWS=ref.CLASSES,ref.FULL_FP32_POLICY,ref.SCENES,ref.VIEWS
BASE=Path(PROJECT)/'runs'/RUN
CLOCK=dict(clock.CLOCK);CHANNEL_SEEDS=dict(ref.CHANNEL_SEEDS);TRAIN_SCENES=ref.TRAIN_SCENES
ARMS=('B00','B10','B01','B11','FULL_AS')
def runtime():
 p=ROOT/'runtime.json';return read(p if p.is_file() else ROOT/LOCAL/'runtime.json')
def rows():return [dict(row_id=a+'-s'+str(SEED),arm=a,model_seed=SEED) for a in ARMS]
def train_rows():return rows()
def parent_config(arm):
 if arm in ('B00','FULL_AS'):return clock.reference_config('REPEAT' if arm=='B00' else 'CONCAT')
 if arm not in ARMS:raise ValueError('Unregistered parent')
 return clock.config(next(r for r in clock.rows() if r['arm']==arm))
def parent_source(arm):return Path(parent_config(arm)['output_root'])
def parent_proof(arm):
 if arm in ('B00','FULL_AS'):
  from experiments.cvs_fullsource_concat20.proof import source
 else:
  from experiments.cvs_clock_factorial20.proof import source
 p=source(parent_config(arm));result=read(p/'training_result.json')
 if result.get('config')!=parent_config(arm) or result.get('actual_steps')!=4440:raise ValueError('Incomplete source-only parent training')
 if read(p/'source_contract.json')['role_ids']!=read(SOURCE)['role_ids']:raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH')
 return p
def config(row):
 if row not in rows():raise ValueError('Only fixed five-row E20+10 amendment')
 c=deepcopy(parent_config(row['arm']));p=c['output_root']
 c.update(schema='F_clock_E20plus10_v1',run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),baseline_reuse=False,epochs=10,label_epochs=0,pseudo_epochs=10,updates=UPDATES,cumulative_epochs=30,cumulative_updates=6660,parent_source=p,parent_epochs=20,parent_selection='all five fixed source-final20;no target result selection',native_scratch_only=False,optimizer_policy='reset AdamW and native cosine10 at original baseLR;not exact trajectory resume',EMA_policy='strict restore own parent EMA;original EMA/gate rules;segment counters/temporal cache reset',ordinary_policy='freeze original E13 schedule using actual parent args',SupCon_policy='cumulative E21..30 original weight0.1/common pair only',operator_epoch_offset=20,operator_step_offset=4440,training_recipe='F+CROSS_RX original U and default head;E20+10/cos10-reset;'+row['arm'])
 c['training_mechanism'].update(expected_L_forward_packets=1136640,original_L_identity_forwards=284160,extra_L_identity_forwards=852480,L_forwards_per_sample=4)
 return c
def validate(c):
 r=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if r is None or c!=config(r):raise ValueError('Fixed E30 parent/data/budget/operator contract differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c);p=ref.make_args(clock.reference_config(),device)
 p.run_id=RUN;p.candidate_id=c['row_id'];p.output_dir=c['output_root'];p.epochs=10;p.label_epochs=0;p.pseudo_epochs=10
 p.from_scratch=False;p.a1_scratch_only=False;p.baseline_ckpt=str(Path(c['parent_source'])/'final_ssdg.pth');p.teacher_ckpt=''
 return p
base_model_config=ref.base_model_config

def validate_parent_checkpoint(cp,c):
 p=parent_config(c['arm'])
 if cp.get('run_id')!=p['run_id'] or cp.get('candidate_id')!=p['row_id'] or cp.get('epoch')!=20 or cp.get('checkpoint_selection')!='final_only' or cp.get('a1_ema_successful_updates')!=4440 or cp.get('ema_model') is None:raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED')
 if cp['args'].get('baseline_ckpt') or cp['args'].get('teacher_ckpt') or cp['args'].get('source_fold_held_rx') is not None:raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED')
 if cp['args'].get('freeze_backbone'):raise ValueError('Default head/body joint parent required')
 return cp
