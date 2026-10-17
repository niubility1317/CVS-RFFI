"""Visible-L batch composition only;three own-parent E20+10 matches,no new IQ Ag."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_clock_factorial30 import design as trained
ROOT=Path(__file__).resolve().parents[2]
RUN='20261018-phase1-F-crossRX-batchStratify-e20plus10-s2026092701-m3-r01';RELEASE='cvs_F_batchStratify_E30_20261018_r01';LOCAL='local_artifacts/F_batchStratify_E30_20261018';OWNER='codex/root/F-batchStratify-E30'
read,write=trained.read,trained.write
PROJECT,SOURCE,CAPSULE,TRUTH,VIEWS_ROOT=trained.PROJECT,trained.SOURCE,trained.CAPSULE,trained.TRUTH,trained.VIEWS_ROOT
AF_RELEASE,SEED,FIT,SCENES,VIEWS,CLASSES,FULL_FP32_POLICY=trained.AF_RELEASE,trained.SEED,trained.FIT,trained.SCENES,trained.VIEWS,trained.CLASSES,trained.FULL_FP32_POLICY
TRAIN_SCENES,CHANNEL_SEEDS=trained.TRAIN_SCENES,trained.CHANNEL_SEEDS
STEPS,UPDATES=222,2220;BASE=Path(PROJECT)/'runs'/RUN
MODES=('ORIGINAL','RANDOM','TX_RX_BALANCED');POOL_BATCHES=6;L_LOADER_SEED=20261018131;U_LOADER_SEED=20261018132;REPARTITION_SEED=20261018121
def runtime():
 p=ROOT/'runtime.json';return read(p if p.is_file() else ROOT/LOCAL/'runtime.json')
def rows():return [dict(row_id=m+'-s'+str(SEED),arm='FULL_AS',model_seed=SEED,batch_mode=m) for m in MODES]
def train_rows():return rows()
def nominal_config():return trained.config(next(r for r in trained.rows() if r['arm']=='FULL_AS'))
def parent_config(arm):
 if arm!='FULL_AS':raise ValueError('Fixed ordinaryAs parent only')
 return trained.parent_config(arm)
parent_source=trained.parent_source
parent_proof=trained.parent_proof;validate_parent_checkpoint=trained.validate_parent_checkpoint;base_model_config=trained.base_model_config
def config(row):
 if row not in rows():raise ValueError('Only registered3 batch modes')
 c=deepcopy(nominal_config());c.update(run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),schema='F_BATCH_REPARTITION_E20PLUS10_V1',model_initialization='strict same original fullsourceAs final20 student+ownEMA;resetAdamW/cos10',parent_selection='Same original fixed budget ordinaryAsE20,not target-ranked',training_recipe='F+CROSS_RX/defaultcos/originalU/ordinary.5As.5/RDrop1/LS.01;E20+10;only Lbatchcomposition differs',batch_policy=dict(pool_batches=6,physical_L_per_pool=768,output_batches=6,physical_L_per_output=128,occurrence_rule='Exact multiset in each pool,including duplicates;no drops/replacement',composition='TXxRX only;day unchanged/retained,not held out',method=row['batch_mode'],L_loader_seed=L_LOADER_SEED,U_loader_seed=U_LOADER_SEED,repartition_seed=REPARTITION_SEED,loader_rng='Common isolated CPU loader streams;U order/hiddenlabels/oldgate/teacher/weakstrong rules retained',prefetch='All3 buffer same6 batches;no extra NN or samples',IQ='Same ordinary policy and original single full practical(raw);batch regroup can change which IQ randomness is assigned to which packet;not exact perpacket-view coupling',loss='Original4 L forwards .5RD(ordinary)+.5RD(As)+absolute.1SupCon;original U losses unchanged'),new_losses=0,new_neural_forwards=0,new_U_rules=0,new_Ag=False,As=True,source_selection='Fixedfinal10/cumulative30;all3 mandatory;sourceVhealth only;no target selection',selection='Allregistered3 final checkpoints;no ranking or score-selective rerun',budget_authorization='Each same originalE20+10=E30;3x2220newupdates;parent20 sharedonce',cost_claim='Each L1136640/studenttrain1703640/teacherU567000 expected;actual eval/smoke and wall recorded;0 extra NN vs original control',scientific_falsification=dict(candidate='TX_RX_BALANCED',references=['ORIGINAL','RANDOM'],each_reference_min_low_urban_gain_pp=.5,each_reference_min_clean_gain_pp=0.,each_reference_min_midurban_gain_pp=0.,each_reference_min_suburban_gain_pp=-.25,interpretation='Allfixed outcomes reported;no automatic seed extension/promotion based on target;not Ag or complete joint proof'))
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed data/batch/parent/E30 matrix differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c);a=trained.make_args(nominal_config(),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];return a
