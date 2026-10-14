from copy import deepcopy
from pathlib import Path
from experiments.cvs_local_order50 import design as p
from experiments.cvs_neighbor_pair50 import design as paired
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,FULL_FP32_POLICY,read,write=(getattr(p,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','FULL_FP32_POLICY','read','write'))
RUN='20261014-phase1-sigtime-pair-oldgate-cos-e50-clean-manysig-s01-m3-r01';RELEASE='cvs_sigtime_pair50_20261014_r01';OWNER='codex/root/sigtime-pair50-20261014';BASE=Path(PROJECT)/'runs'/RUN
SEEDS=(2026092701,);ARMS=('OLD_GATE_COS50','SIG_TIME','SIG_PAIR');BASELINE_ARM=ARMS[0];PRIMARY=ARMS[-1];FEATURE_CONTROL='SIG_TIME';FEATURE_PARAMETER_COUNT=195041;VERIFY_INITIAL_EVAL_EQUAL=True
PAIRS=[(PRIMARY,'SIG_TIME'),(PRIMARY,BASELINE_ARM),('SIG_TIME',BASELINE_ARM)]
READOUTS,VIEWS,QUERY_COUNT,EPOCHS,LABEL_EPOCHS,PSEUDO_EPOCHS,STEPS,UPDATES,BASELINE_EVAL_RUN,STYLE_LAYERS=(getattr(p,k) for k in ('READOUTS','VIEWS','QUERY_COUNT','EPOCHS','LABEL_EPOCHS','PSEUDO_EPOCHS','STEPS','UPDATES','BASELINE_EVAL_RUN','STYLE_LAYERS'))
baseline_config=p.baseline_config
PARAMETERS=('id_backbone.encoder.core.id_backbone.t1.pw.weight','id_backbone.encoder.core.id_backbone.t2.pw.weight','id_backbone.encoder.sequence_pool.views.0.local_projection.weight');DIAGNOSTIC_EPOCHS=(33,34,50)
def rows():return [dict(row_id=f'{a}-s{s}',arm=a,model_seed=s) for s in SEEDS for a in ARMS]
def training_rows():return [r for r in rows() if r['arm']!=BASELINE_ARM]
def config(row):
 if row not in rows():raise ValueError('Unregistered fixedSIG pair row')
 skeleton=row if row['arm']==BASELINE_ARM else dict(row,arm='SIG_TIME',row_id=f"SIG_TIME-s{row['model_seed']}")
 c=deepcopy(p.config(skeleton));c.pop('local_order');c.update(run_id=RUN,**row,output_root=(BASE/row['row_id']/'source').as_posix(),launch_owner=OWNER)
 sampling=deepcopy(paired.config(paired.training_rows()[0])['source_neighbor_sampling']);sampling.update(enabled=row['arm']!=BASELINE_ARM,mode='PAIRS' if row['arm']==PRIMARY else 'RANDOM');c['source_neighbor_sampling']=sampling
 c['sampling_observation']=dict(actual_gradient_epochs=list(DIAGNOSTIC_EPOCHS),batches_per_epoch=222,gradient_parameters=list(PARAMETERS),vectors=['main_L_CE','weighted_U_CE','weighted_U_CE_entropy'],loss_weights=dict(U_CE=.16,U_entropy=.01),extra_model_forwards=0,extra_optimizer_updates=0,head_fit_updates=0,optimizer='unchangednativeAdamW',scope='Actual separateL128/U256 graphs. No cumulative AdamW contribution claim; original losses returned unchanged')
 c['hypothesis_basis']='User fixesSIG_TIME andprioritizes training/DG/SSL. Report31957f78 recommends standaloneSIG_TIME+PAIR+uniformUCE vsrandom. ExistingPAIRsourcegateprecision/coverage evidence andoriginalfull3candidate preserved; this is integration reference, not newalgorithm or 84-coreclaim.'
 c['source_falsification']=dict(final_L_CE_max=.5,minimum_paired_U_acceptance=.25,source_feature_difference_min=1e-6,gradient_batches=666,source_only=True,expansion='Both source finalCE<=.5, sampler17x56700/exact11100updates/actualfeatureboundary &paired-minus-randomL720>=1e-6; pairedUaccepted/963900>=.25. Samefixed02/03, no target ranking.')
 c['goal']=dict(primary=PRIMARY,clean_three_seed_mean=.84,first_seed_is_not_goal_completion=True,feedback_to_training=False,integration_not_new_core_algorithm=True);return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Frozen SIG_TIME pair integration differs')
def make_args(c,device='cuda:0'):
 validate(c)
 if c['baseline_reuse']:raise ValueError('Do not retrain historical baseline')
 a=p.p.parent.first.old.make_args(baseline_config(c),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];return a
