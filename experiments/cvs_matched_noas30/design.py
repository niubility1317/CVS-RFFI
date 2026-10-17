"""Mandatory no-As reference;same source roles and E30 rules,not a new candidate."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_clock_factorial30 import design as trained
from experiments.cvs_view_pcgrad30 import design as view
ROOT=Path(__file__).resolve().parents[2]
RUN='20261018-phase1-F-crossRX-matchedNoAs-e20plus10-s2026092701-m1-r01';RELEASE='cvs_F_matchedNoAs_E30_20261018_r01';LOCAL='local_artifacts/F_matchedNoAs_E30_20261018';OWNER='codex/root/F-matchedNoAs-E30'
read,write=trained.read,trained.write
PROJECT,SOURCE,CAPSULE,TRUTH,VIEWS_ROOT=trained.PROJECT,trained.SOURCE,trained.CAPSULE,trained.TRUTH,trained.VIEWS_ROOT
AF_RELEASE,SEED,FIT,SCENES,VIEWS,CLASSES,FULL_FP32_POLICY=trained.AF_RELEASE,trained.SEED,trained.FIT,trained.SCENES,trained.VIEWS,trained.CLASSES,trained.FULL_FP32_POLICY
TRAIN_SCENES,CHANNEL_SEEDS=trained.TRAIN_SCENES,trained.CHANNEL_SEEDS
STEPS,UPDATES=222,2220;BASE=Path(PROJECT)/'runs'/RUN
MODES=('ORIGINAL','MEASURE','PROJECT');POOL_BATCHES=6;L_LOADER_SEED=view.L_LOADER_SEED;U_LOADER_SEED=view.U_LOADER_SEED;REPARTITION_SEED=view.REPARTITION_SEED
parent_config,parent_source,parent_proof,validate_parent_checkpoint,base_model_config=trained.parent_config,trained.parent_source,trained.parent_proof,trained.validate_parent_checkpoint,trained.base_model_config
def runtime():
 p=ROOT/'runtime.json';return read(p if p.is_file() else ROOT/LOCAL/'runtime.json')
def rows():return [dict(row_id='B00-s'+str(SEED),arm='B00',model_seed=SEED,batch_mode='ORIGINAL',gradient_mode='MEASURE')]
def train_rows():return rows()
def nominal_config():return trained.config(next(r for r in trained.rows() if r['arm']=='B00'))
def config(row):
 if row not in rows():raise ValueError('Only mandatory fixed B00 reference')
 c=deepcopy(nominal_config());v=view.config(view.rows()[1]);c.update(run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),schema='F_MATCHED_NO_AS_E30_V1',model_initialization='strict original fullsourceNoAs REPEAT final20 student+ownEMA;resetAdamW/cos10;same original scratchmodelseed as As parent',parent_selection='Original preregistered REPEAT20 noAs control,not selected by target results',training_recipe='F+CROSS_RX/defaultcos/originalU/ordinary RD1/RDrop1/LS.01;noAs E20+10;MEASURE twoVJP no gradient edit',batch_policy=v['batch_policy'],gradient_policy=dict(v['gradient_policy'],mode='MEASURE',satellite='Inactive-view control repeats same generated ordinary IQ;no practical is generated',rule='Measure both halfweighted ordinary RDrop pair gradients;do not edit native total gradient'),new_losses=0,new_neural_forwards=0,new_U_rules=0,new_Ag=False,As=False,source_selection='Mandatory final10/cumulative30,no source or target selection',selection='One fixed reference;not candidate promotion',budget_authorization='Original matching noAsE20 parent+10=E30;one new2220update row;no extension to live As rows',cost_claim='4L NN and2extraVJP eachstep matches As PROJECT/MEASURE;no physical As generation;teacher/eval/prediction/wall separate',reference_for_run='20261018-phase1-F-crossRX-viewPCGrad-e20plus10-s2026092701-m3-r02',clean_loss_mass='Common ordinary pair.5 plus inactive ordinary pair.5=1;As rows common ordinary.5 plus As.5;totalL1 andSupCon.1 explicit',scientific_falsification=None)
 return c
def validate(c):
 if c!=config(rows()[0]):raise ValueError('Fixed mandatory reference contract differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c);a=trained.make_args(nominal_config(),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];return a
