"""Four fixed full-source E20+10 rows: optimizer recipe x existing practical As."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_clock_factorial30 import design as previous
ROOT=Path(__file__).resolve().parents[2]
RUN='20261018-phase1-F-crossRX-viewMoment-e20plus10-s2026092701-m4-r01';RELEASE='cvs_F_viewMoment_E30_20261018_r01';LOCAL='local_artifacts/F_view_second_moment_20261018';OWNER='codex/root/F-viewMoment-E30'
read,write=previous.read,previous.write
PROJECT,SOURCE,CAPSULE,TRUTH,VIEWS_ROOT=previous.PROJECT,previous.SOURCE,previous.CAPSULE,previous.TRUTH,previous.VIEWS_ROOT
AF_RELEASE,SEED,FIT,SCENES,VIEWS,CLASSES,FULL_FP32_POLICY=previous.AF_RELEASE,previous.SEED,previous.FIT,previous.SCENES,previous.VIEWS,previous.CLASSES,previous.FULL_FP32_POLICY
TRAIN_SCENES,CHANNEL_SEEDS=previous.TRAIN_SCENES,previous.CHANNEL_SEEDS
STEPS,UPDATES=222,2220;BASE=Path(PROJECT)/'runs'/RUN
POOL_BATCHES=6;L_LOADER_SEED=20261018131;U_LOADER_SEED=20261018132;REPARTITION_SEED=20261018121
BATCH_POLICY=dict(pool_batches=6,physical_L_per_pool=768,output_batches=6,physical_L_per_output=128,method='ORIGINAL',L_loader_seed=L_LOADER_SEED,U_loader_seed=U_LOADER_SEED,repartition_seed=REPARTITION_SEED,occurrence_rule='All four identical buffered original input order and isolated loader streams')
parent_config,parent_source,parent_proof,validate_parent_checkpoint,base_model_config=previous.parent_config,previous.parent_source,previous.parent_proof,previous.validate_parent_checkpoint,previous.base_model_config
MODES=('MEASURE','VIEW_SECOND')
def runtime():
 p=ROOT/'runtime.json';return read(p if p.is_file() else ROOT/LOCAL/'runtime.json')
def rows():
 return [dict(row_id=key+'-s'+str(SEED),arm=arm,model_seed=SEED,batch_mode='ORIGINAL',moment_mode=mode) for key,arm,mode in [('NOAS_MEASURE','B00','MEASURE'),('AS_MEASURE','FULL_AS','MEASURE'),('NOAS_VIEW','B00','VIEW_SECOND'),('AS_VIEW','FULL_AS','VIEW_SECOND')]]
def train_rows():return rows()
def nominal_config(arm):return previous.config(next(r for r in previous.rows() if r['arm']==arm))
def config(row):
 if row not in rows():raise ValueError('Only fixed four optimizer/As rows')
 c=deepcopy(nominal_config(row['arm']));c.update(run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),schema='F_VIEW_SECOND_MOMENT_E20PLUS10_V1',model_initialization='Own original fullsource REPEAT20 for NoAs / CONCAT20 for As,strict student and ownEMA;AdamW/cos10 reset;not shared treatment weights',parent_selection='Original fixed final20 perarm;not target-ranked',training_recipe='F+CROSS_RX/defaultcos/originalU/RDrop1/LS.01;ordinary.5 plus ordinary-or-practical.5;SupCon.1;only AdamW second moment differs',batch_policy=deepcopy(BATCH_POLICY),moment_policy=dict(mode=row['moment_mode'],weighted_views=['.5 ordinaryRD','.5 secondaryRD'],native_gradient='g_total=gO+gS+gSupCon+gU,original physical clip5',variance='delta=(gO-gS)^2 in stored parameter coordinates,scaled by the same clip factor squared',second='beta2*v+(1-beta2)*(g_total_postclip^2+delta_postclip) for VIEW_SECOND;native for MEASURE',first='unchanged native first moment;original groupedSinc lr/epsilon/physicaldecay unchanged',coefficient='Exactly1 from equal two-view mixture,not tuned',limits='Not unbiased totalgradient variance or an AdamW bugfix;no stepdescent guarantee',extra_VJP_per_step=2,total_L_forwards=4,extra_neural_forwards=0,objective_unchanged=True,new_deployment_parameters=0,teacher_U='Original U/gates/ownEMA unchanged;no U VJP or replay'),new_losses=0,new_neural_forwards=0,new_U_rules=0,new_Ag=False,As=row['arm']=='FULL_AS',source_selection='Every fixed final10/cumulative30;source qualification before target metric read;allrows mandatory',selection='No target ranking,parameter changes or selective rerun',budget_authorization='Each ownE20+10=E30;4x2220 newupdates;2uniqueE20parents counted once',cost_claim='All4 exact4Lforwards+2VJP;no additional IQ;U/teacher unchanged;optimizer arithmetic and actual time counted separately',scientific_falsification=dict(candidate='AS_VIEW',reference='AS_MEASURE',source='V1800/full5RX/2fixedpracticaldraws;beforetarget',min_clean_gain_pp=0.,min_each_draw_midurban_gain_pp=0.,min_each_draw_lowurban_gain_pp=.5,min_each_suburban_gain_pp=-.25,min_NOAS_VIEW_clean_vs_NOAS_MEASURE_pp=-.25,float_tolerance_pp=1e-10,on_fail='Close fixed view-second-moment recipe;no coefficient/parameter-subset/parent/seed/budget rescue',on_pass='Source qualification only;not IQAg or joint completion,no automatic target-guided expansion'),REJECTED_EXTRA_GATE='No separate design approval,seal or additional review')
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed data/parent/E30/moment matrix differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c);a=previous.make_args(nominal_config(c['arm']),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];return a
