"""L-only same-IQ SAM with matched replay/NoAs;one joint default-head model."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_clock_factorial30 import design as trained
ROOT=Path(__file__).resolve().parents[2]
RUN='20261018-phase1-F-crossRX-LSAM-e20plus10-s2026092701-m3-r01';RELEASE='cvs_F_LSAM_E30_20261018_r01';LOCAL='local_artifacts/F_LSAM_E30_20261018';OWNER='codex/root/F-LSAM-E30'
read,write=trained.read,trained.write
PROJECT,SOURCE,CAPSULE,TRUTH,VIEWS_ROOT=trained.PROJECT,trained.SOURCE,trained.CAPSULE,trained.TRUTH,trained.VIEWS_ROOT
AF_RELEASE,SEED,FIT,SCENES,VIEWS,CLASSES,FULL_FP32_POLICY=trained.AF_RELEASE,trained.SEED,trained.FIT,trained.SCENES,trained.VIEWS,trained.CLASSES,trained.FULL_FP32_POLICY
TRAIN_SCENES,CHANNEL_SEEDS=trained.TRAIN_SCENES,trained.CHANNEL_SEEDS
STEPS,UPDATES=222,2220;BASE=Path(PROJECT)/'runs'/RUN
POOL_BATCHES=6;L_LOADER_SEED=20261018131;U_LOADER_SEED=20261018132;REPARTITION_SEED=20261018121
BATCH_POLICY=dict(pool_batches=6,physical_L_per_pool=768,output_batches=6,physical_L_per_output=128,method='ORIGINAL',L_loader_seed=L_LOADER_SEED,U_loader_seed=U_LOADER_SEED,repartition_seed=REPARTITION_SEED,occurrence_rule='All three identical buffered original input order and isolated loader streams')
RHO=.05
parent_config,parent_source,parent_proof,validate_parent_checkpoint,base_model_config=trained.parent_config,trained.parent_source,trained.parent_proof,trained.validate_parent_checkpoint,trained.base_model_config
def runtime():
 p=ROOT/'runtime.json';return read(p if p.is_file() else ROOT/LOCAL/'runtime.json')
def rows():return [dict(row_id=k+'-s'+str(SEED),arm=arm,model_seed=SEED,batch_mode='ORIGINAL',update_mode=mode) for k,arm,mode in [('NOAS_REPLAY','B00','REPLAY'),('AS_REPLAY','FULL_AS','REPLAY'),('AS_LSAM','FULL_AS','LSAM')]]
def train_rows():return rows()
def nominal_config(arm):return trained.config(next(r for r in trained.rows() if r['arm']==arm))
def config(row):
 if row not in rows():raise ValueError('Only fixed3 LSAM rows')
 c=deepcopy(nominal_config(row['arm']));c.update(run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),schema='F_LSAM_E20PLUS10_V1',model_initialization='Own original same-contract REPEAT20 for NoAs/CONCAT20 for As,strict student+ownEMA;resetAdamW/cos10;same initialization rule,not identical treatment parents',parent_selection='Original preregistered final20 perarm;not target-selected',training_recipe='F+CROSS_RX/defaultcos/originalU/RDrop1/LS.01;commonordinary.5 plus ordinary-or-fullAs.5;SupCon.1;E20+10;only supervised L update organization differs',batch_policy=deepcopy(BATCH_POLICY),update_policy=dict(mode=row['update_mode'],radius=RHO if row['update_mode']=='LSAM' else 0.,question='Does bounded L-objective parameter-neighborhood training improve practical adaptation and protect clean versus exact same-IQ replay?',perturbation='Vanilla SAM radius.05 in actual stored F coordinates including CENTER25K q;not physical-Hz invariant or an IQ perturbation',L_objective='.5ordinaryRD+.5secondaryRD+.1originalshallowmiddlecrossRXSupCon',rule='Capture gL;native backward supplies gtotal. At original preclip:replay same4 inputs and each exact RNG state at theta+rho*gL/||gL||;restore weights exactly;LSAM replaces gL by replaygL in gtotal;REPLAY leaves native gradient unchanged. Original clip5/AdamW/ownEMA eachonce',U='Native U objective/teacher/gates/forwards and gradient contribution left unchanged;no U replay or truth',IQ='Generate ordinary/fullpractical once;reuse both RD pairs and replay;no new As or IQ draws',rng='Capture at each actual encoder.features entry;restore all global RNG after replay;no redraw for U or next step',new_deployment_parameters=0,extra_L_forwards=4,extra_VJP_per_step=2,total_L_forwards=8,actual_finite_step_guarantee=False),new_losses=0,new_neural_forwards=4,new_U_rules=0,new_Ag=False,As=row['arm']=='FULL_AS',source_selection='All3 fixedfinal10/cumulative30;no checkpoint ranking;source qualification completes before same-row fixed testing',selection='All3 mandatory final checkpoints;no target selection,hyperparameter update or selective rerun',budget_authorization='Each originalE20+10 cumulativeE30;3x2220newupdates;two unique parentE20 costs recorded separately',cost_claim='All3 8L NN/2extraVJP;ordinaryAs generatedonce per128;original U/teacher unchanged. Report parent,new,total updates/NN/VJP/teacher/wall and synthetic separately',scientific_falsification=dict(candidate='AS_LSAM',reference='AS_REPLAY',source='Fixed originalV1800 all5RX/6TX/3days,2draws;no updates;before target read',min_clean_gain_pp=0.,min_midurban_gain_pp=0.,min_lowurban_gain_pp=.5,min_each_suburban_gain_pp=-.25,on_fail='Close fixed L-SAM radius/coordinates/parent/seed/recipe;no radius/subset/optimizer/budget rescue',on_pass='Only source qualification,not Ag or joint proof;no automatic multi-seed from target'),REJECTED_EXTRA_GATE='Generic design approval/spec/plan gates exceed current authorized minimal experiment workflow;not imposed')
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed L-SAM data/parent/budget/update matrix differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c);a=trained.make_args(nominal_config(c['arm']),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];return a
