from copy import deepcopy
from pathlib import Path
from experiments.cvs_local_order50 import design as p
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,FULL_FP32_POLICY,read,write=(getattr(p,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','FULL_FP32_POLICY','read','write'))
RUN='20261014-phase1-signature-order-oldgate-cos-e50-clean-manysig-s01-m4-r01';RELEASE='cvs_signature_order50_20261014_r01';OWNER='codex/root/signature-order50-20261014';BASE=Path(PROJECT)/'runs'/RUN
SEEDS=(2026092701,);ARMS=('OLD_GATE_COS50','SIG_TIME','SIG_RAW_ORDER','SIG_DYNAMIC_ORDER');BASELINE_ARM=ARMS[0];PRIMARY=ARMS[-1];FEATURE_CONTROL='SIG_TIME';FEATURE_PARAMETER_COUNT=195041;VERIFY_INITIAL_EVAL_EQUAL=True
PAIRS=[(PRIMARY,'SIG_RAW_ORDER'),(PRIMARY,'SIG_TIME'),(PRIMARY,BASELINE_ARM),('SIG_RAW_ORDER','SIG_TIME'),('SIG_TIME',BASELINE_ARM)]
READOUTS,VIEWS,QUERY_COUNT,EPOCHS,LABEL_EPOCHS,PSEUDO_EPOCHS,STEPS,UPDATES,BASELINE_EVAL_RUN,STYLE_LAYERS=(getattr(p,k) for k in ('READOUTS','VIEWS','QUERY_COUNT','EPOCHS','LABEL_EPOCHS','PSEUDO_EPOCHS','STEPS','UPDATES','BASELINE_EVAL_RUN','STYLE_LAYERS'))
baseline_config=p.baseline_config
COHORT_SEED=p.COHORT_SEED;DIAGNOSTIC_EPOCHS=p.DIAGNOSTIC_EPOCHS
PARAMETERS=('id_backbone.encoder.core.id_backbone.t1.pw.weight','id_backbone.encoder.core.id_backbone.t2.pw.weight','id_backbone.encoder.sequence_pool.views.0.local_projection.weight')
LEVELS=('SIG_RAW','SIG_DYNAMIC')
def rows():return [dict(row_id=f'{a}-s{s}',arm=a,model_seed=s) for s in SEEDS for a in ARMS]
def training_rows():return [r for r in rows() if r['arm']!=BASELINE_ARM]
def config(row):
 if row not in rows():raise ValueError('Unregistered signature order row')
 skeleton=row if row['arm']==BASELINE_ARM else dict(row,arm='SIG_TIME',row_id=f"SIG_TIME-s{row['model_seed']}")
 c=deepcopy(p.config(skeleton));c.pop('local_order');c.update(run_id=RUN,**row,output_root=(BASE/row['row_id']/'source').as_posix(),launch_owner=OWNER)
 c['signature_order']=dict(mode='ZERO' if row['arm']=='SIG_TIME' else 'SIG_RAW' if row['arm']=='SIG_RAW_ORDER' else 'SIG_DYNAMIC' if row['arm']==PRIMARY else 'historical',source_role='L_s',cohort_seed=COHORT_SEED,cohort_packets=90,cohorts=70,every_steps=4,extra_batches_per_epoch=55,extra_batches=2750,extra_source_packet_forwards=247500,margin=.1,coefficient=.1,warmup_epochs=5,ramp_epochs=10,view='actual deployedSIG112; dynamic108 removes only4 input-independent time-step slots before L2eps1e-4; time-area coordinates retained; deployed network unchanged',loss='all1800 same-day crossRX identity triples incl satisfiedzero, mean denominator fixed1800; no fitted projection/head',extra_mode='eval with gradients; restore module modes and CPU/GPU RNG; identical extraL90forwards all3arms',diagnostic_epochs=list(DIAGNOSTIC_EPOCHS),gradient_objects=list(PARAMETERS),no_U_relations=True,no_extra_head=True)
 c['hypothesis_basis']='AllL6300 source signature diagnostic: time-step4constant energyfraction17.6511%, raw/dynamic identityrelations and actualSIGprojection gradients measured. Separate supervision position and constant normalization; fixed existingmargin/coefficient/calendar, no target-selected weights or hierarchy/PAIR/weight stacking.'
 c['source_falsification']=dict(final_L_CE_max=.5,feature_difference_min=1e-6,actual_extra_batches=2750,gradient_batches=220,all_candidate_gradient_parameters_nonzero_at=[17,33,50],source_only=True,expansion='Both candidates must pass same source rules; expand both fixed02/03 unchanged, no target ranking')
 c['goal']=dict(primary=PRIMARY,clean_three_seed_mean=.84,first_seed_is_not_goal_completion=True,feedback_to_training=False);return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Frozen signature order recipe differs')
def make_args(c,device='cuda:0'):
 validate(c)
 if c['baseline_reuse']:raise ValueError('Do not retrain historical baseline')
 a=p.p.parent.first.old.make_args(baseline_config(c),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];return a
