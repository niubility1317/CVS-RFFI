"""Supervised ordinaryAs gradient organization;three same-parent E20+10 matches,no new IQAg."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_clock_factorial30 import design as trained
ROOT=Path(__file__).resolve().parents[2]
RUN='20261018-phase1-F-crossRX-viewPCGrad-e20plus10-s2026092701-m3-r02';RELEASE='cvs_F_viewPCGrad_E30_20261018_r02';LOCAL='local_artifacts/F_viewPCGrad_E30_20261018_r02';OWNER='codex/root/F-viewPCGrad-E30'
read,write=trained.read,trained.write
PROJECT,SOURCE,CAPSULE,TRUTH,VIEWS_ROOT=trained.PROJECT,trained.SOURCE,trained.CAPSULE,trained.TRUTH,trained.VIEWS_ROOT
AF_RELEASE,SEED,FIT,SCENES,VIEWS,CLASSES,FULL_FP32_POLICY=trained.AF_RELEASE,trained.SEED,trained.FIT,trained.SCENES,trained.VIEWS,trained.CLASSES,trained.FULL_FP32_POLICY
TRAIN_SCENES,CHANNEL_SEEDS=trained.TRAIN_SCENES,trained.CHANNEL_SEEDS
STEPS,UPDATES=222,2220;BASE=Path(PROJECT)/'runs'/RUN
MODES=('ORIGINAL','MEASURE','PROJECT');POOL_BATCHES=6;L_LOADER_SEED=20261018131;U_LOADER_SEED=20261018132;REPARTITION_SEED=20261018121
def runtime():
 p=ROOT/'runtime.json';return read(p if p.is_file() else ROOT/LOCAL/'runtime.json')
def rows():return [dict(row_id=m+'-s'+str(SEED),arm='FULL_AS',model_seed=SEED,batch_mode='ORIGINAL',gradient_mode=m) for m in MODES]
def train_rows():return rows()
def nominal_config():return trained.config(next(r for r in trained.rows() if r['arm']=='FULL_AS'))
def parent_config(arm):
 if arm!='FULL_AS':raise ValueError('Fixed ordinaryAs parent only')
 return trained.parent_config(arm)
parent_source=trained.parent_source
parent_proof=trained.parent_proof;validate_parent_checkpoint=trained.validate_parent_checkpoint;base_model_config=trained.base_model_config
def config(row):
 if row not in rows():raise ValueError('Only registered3 batch modes')
 c=deepcopy(nominal_config());c.update(run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),schema='F_VIEW_PCGRAD_E20PLUS10_V1',model_initialization='strict same original fullsourceAs final20 student+ownEMA;resetAdamW/cos10',parent_selection='Same original fixed budget ordinaryAsE20,not target-ranked',training_recipe='F+CROSS_RX/defaultcos/originalU/ordinary.5As.5/RDrop1/LS.01;E20+10;only supervised-view gradient organization differs',batch_policy=dict(pool_batches=6,physical_L_per_pool=768,output_batches=6,physical_L_per_output=128,method='ORIGINAL',L_loader_seed=L_LOADER_SEED,U_loader_seed=U_LOADER_SEED,repartition_seed=REPARTITION_SEED,occurrence_rule='All three identical buffered original input order and isolated loader streams'),gradient_policy=dict(mode=row['gradient_mode'],ordinary='.5 existing ordinary RDrop objective',satellite='.5 existing full practical RDrop objective;all128 legalL,not confidence filtered',remaining='Original SupCon.1 and U losses unchanged',coordinate='Physical Hz gradients for CENTER25K Sinc divided by25000;all other original parameter coordinates,including defaulthead',rule='If gO dot gS<0,replace gS by gS-(gO dot gS)/(gO dot gO)*gO before original physical clip5 and AdamW;one-sided clean preference',scope='All and only original jointly trainable parameters',new_weights=0,actual_AdamW_descent_guarantee=False,extra_VJP_per_step=0 if row['gradient_mode']=='ORIGINAL' else 2,extra_neural_forwards=0),new_losses=0,new_neural_forwards=0,new_U_rules=0,new_Ag=False,As=True,source_selection='Fixedfinal10/cumulative30;all3 mandatory;sourceVhealth only;no target selection',selection='Allregistered3 final checkpoints;no ranking or score-selective rerun',budget_authorization='Each same originalE20+10=E30;3x2220newupdates;parent20 sharedonce',cost_claim='No extra NN;MEASURE/PROJECT each2 extra supervised VJPs per update,wall and backward cost separate',scientific_falsification=dict(candidate='PROJECT',references=['ORIGINAL','MEASURE'],each_reference_min_low_urban_gain_pp=.5,each_reference_min_clean_gain_pp=0.,each_reference_min_midurban_gain_pp=0.,each_reference_min_suburban_gain_pp=-.25,interpretation='Fixed single-seed efficacy falsification only;alloutcomes;no target-guided tuning/seed extension;not IQAg or joint completion'))
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed data/batch/parent/E30 matrix differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c);a=trained.make_args(nominal_config(),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];return a
