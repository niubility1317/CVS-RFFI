from copy import deepcopy
from pathlib import Path
from experiments.cvs_trusted_weight50 import design as p
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,FULL_FP32_POLICY,read,write=(getattr(p,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','FULL_FP32_POLICY','read','write'))
RUN='20261014-phase1-local-order-oldgate-cos-e50-clean-manysig-s01-m4-r01';RELEASE='cvs_local_order50_20261014_r01';OWNER='codex/root/local-order50-20261014';BASE=Path(PROJECT)/'runs'/RUN
SEEDS=(2026092701,);ARMS=('OLD_GATE_COS50','SIG_TIME','LOCAL_SUPCON','LOCAL_ORDER');BASELINE_ARM=ARMS[0];PRIMARY=ARMS[-1];FEATURE_CONTROL='SIG_TIME';FEATURE_PARAMETER_COUNT=195041;VERIFY_INITIAL_EVAL_EQUAL=True
PAIRS=[(PRIMARY,'LOCAL_SUPCON'),(PRIMARY,'SIG_TIME'),(PRIMARY,BASELINE_ARM),('LOCAL_SUPCON','SIG_TIME'),('SIG_TIME',BASELINE_ARM)]
READOUTS,VIEWS,QUERY_COUNT,EPOCHS,LABEL_EPOCHS,PSEUDO_EPOCHS,STEPS,UPDATES,BASELINE_EVAL_RUN,STYLE_LAYERS=(getattr(p,k) for k in ('READOUTS','VIEWS','QUERY_COUNT','EPOCHS','LABEL_EPOCHS','PSEUDO_EPOCHS','STEPS','UPDATES','BASELINE_EVAL_RUN','STYLE_LAYERS'))
baseline_config=p.baseline_config
COHORT_SEED=2026210712;DIAGNOSTIC_EPOCHS=(1,17,33,50)
def rows():return [dict(row_id=f'{a}-s{s}',arm=a,model_seed=s) for s in SEEDS for a in ARMS]
def training_rows():return [r for r in rows() if r['arm']!=BASELINE_ARM]
def config(row):
 if row not in rows():raise ValueError('Unregistered local order row')
 baseline=row['arm']==BASELINE_ARM;skeleton=row if baseline else dict(row,arm='SIG_TIME',row_id=f"SIG_TIME-s{row['model_seed']}")
 c=deepcopy(p.config(skeleton));c.pop('trusted_weight');c.update(run_id=RUN,**row,output_root=(BASE/row['row_id']/'source').as_posix(),launch_owner=OWNER)
 c['local_order']=dict(mode='ZERO' if row['arm']=='SIG_TIME' else 'SUPCON' if row['arm']=='LOCAL_SUPCON' else 'ORDER' if row['arm']==PRIMARY else 'historical',layers=['t1','t2'],local_view='4 ordered adaptive average temporal bins on existing postDropout output; perpacketL2eps1e-4; no fitted projection',source_role='L_s',cohort_seed=COHORT_SEED,cohort_packets=90,cohorts=70,cohort_order='fixed rank within TX/RX/day, cycle70 cohorts independently of modelseed, no target or Utruth',every_steps=4,extra_batches_per_epoch=55,extra_batches=2750,extra_source_packet_forwards=247500,all_arms_same_extra_forward=True,extra_mode='eval with parameter gradients; restore module modes and CPU/GPU RNG; classW receives only original joint training',margin=.1,temperature=.2,coefficient=.1,warmup_epochs=5,ramp_epochs=10,loss='mean t1/t2; ORDER all1800legaltriplets incl satisfiedzeros; SUPCON360 anchor/referenceRX groups withsameTXpositive+5otherTXnegative',diagnostic_epochs=list(DIAGNOSTIC_EPOCHS),diagnostic_batches='all55 scheduled extra Lcohorts perdeclared epoch; actual mainL128CE versus separateL90aux',gradient_objects=['t1.pw.weight','t2.pw.weight','t3.pw.weight'],no_extra_head=True,no_U_relations=True)
 c['hypothesis_basis']='Report31957f78 plus fullL6300source layer order; fixedm.1. Direct shallow/middle semantic supervision vs matched source-reference SupCon, no hierarchy/PAIR/weight stacking or target selection.'
 c['source_falsification']=dict(final_L_CE_max=.5,source_added_feature_difference_min=1e-6,actual_extra_batches=2750,gradient_epoch_batches=220,source_only=True,description='All3finalLCE<=.5; actualupdate/export/Windependence/identicalinitialeval; ORDER-SIGL720zL2>=1e-6; exactextra/gradientbudget; ORDER t1/t2 effectiveauxnonzero at17/33/50; no target score used for expansion.')
 c['goal']=dict(primary=PRIMARY,clean_three_seed_mean=.84,first_seed_is_not_goal_completion=True,feedback_to_training=False);return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Frozen local order recipe differs')
def make_args(c,device='cuda:0'):
 validate(c)
 if c['baseline_reuse']:raise ValueError('Do not retrain historical baseline')
 a=p.parent.first.old.make_args(baseline_config(c),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];return a
