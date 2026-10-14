from copy import deepcopy
from pathlib import Path
from experiments.cvs_signature_order50 import design as p
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,FULL_FP32_POLICY,read,write=(getattr(p,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','FULL_FP32_POLICY','read','write'))
RUN='20261014-phase1-backbone-fishr-oldgate-cos-e50-clean-manysig-s01-m4-r01';RELEASE='cvs_backbone_fishr50_20261014_r01';OWNER='codex/root/backbone-fishr50-20261014';BASE=Path(PROJECT)/'runs'/RUN
SEEDS=(2026092701,);ARMS=('OLD_GATE_COS50','SIG_TIME','BONE_RAW_FISHR','BONE_COND_FISHR');BASELINE_ARM=ARMS[0];PRIMARY=ARMS[-1];FEATURE_CONTROL='SIG_TIME';FEATURE_PARAMETER_COUNT=195041;VERIFY_INITIAL_EVAL_EQUAL=True
PAIRS=[(PRIMARY,'BONE_RAW_FISHR'),(PRIMARY,'SIG_TIME'),(PRIMARY,BASELINE_ARM),('BONE_RAW_FISHR','SIG_TIME'),('SIG_TIME',BASELINE_ARM)]
READOUTS,VIEWS,QUERY_COUNT,EPOCHS,LABEL_EPOCHS,PSEUDO_EPOCHS,STEPS,UPDATES,BASELINE_EVAL_RUN,STYLE_LAYERS=(getattr(p,k) for k in ('READOUTS','VIEWS','QUERY_COUNT','EPOCHS','LABEL_EPOCHS','PSEUDO_EPOCHS','STEPS','UPDATES','BASELINE_EVAL_RUN','STYLE_LAYERS'))
baseline_config=p.baseline_config
COHORT_SEED=p.COHORT_SEED;DIAGNOSTIC_EPOCHS=(1,17,33,50)
PARAMETERS=p.PARAMETERS
FISHR_PARAMETERS=PARAMETERS[:2]
LEVELS=('RAW','TX_DAY_CENTERED')
RX=(1,3,4,6,8);DAYS=(1,2,3)
INTERVAL=16;EXTRA_PER_EPOCH=13;EXTRA_BATCHES=650;COHORTS=35;COHORT_PACKETS=180;COEFFICIENT=.01

def rows():return [dict(row_id=f'{a}-s{s}',arm=a,model_seed=s) for s in SEEDS for a in ARMS]
def training_rows():return [r for r in rows() if r['arm']!=BASELINE_ARM]
def config(row):
 if row not in rows():raise ValueError('Unregistered backbone Fishr row')
 skeleton=row if row['arm']==BASELINE_ARM else dict(row,arm='SIG_TIME',row_id=f"SIG_TIME-s{row['model_seed']}")
 c=deepcopy(p.config(skeleton));c.pop('signature_order');c.update(run_id=RUN,**row,output_root=(BASE/row['row_id']/'source').as_posix(),launch_owner=OWNER)
 c['backbone_fishr']=dict(mode='ZERO' if row['arm']=='SIG_TIME' else 'RAW' if row['arm']=='BONE_RAW_FISHR' else 'TX_DAY_CENTERED' if row['arm']==PRIMARY else 'historical',source_role='L_s',cohort_seed=COHORT_SEED,cohort_packets=COHORT_PACKETS,cohorts=COHORTS,every_steps=INTERVAL,extra_batches_per_epoch=EXTRA_PER_EPOCH,extra_batches=EXTRA_BATCHES,extra_source_packet_forwards=EXTRA_BATCHES*COHORT_PACKETS,coefficient=COEFFICIENT,warmup_epochs=5,ramp_epochs=10,gradient_objects=list(FISHR_PARAMETERS),diagnostic_gradient_objects=list(PARAMETERS),diagnostic_epochs=list(DIAGNOSTIC_EPOCHS),per_sample_gradient='actual default CE smoothing.01 derivatives wrt t1PW3456 andt2PW6912; create_graph=True',domains=list(RX),domain_packets=36,TX_day_RX_packets=2,variance_denominator='n-1=35 within RX',penalty='mean of2 layerwise meanRX squared variance-vector discrepancy / squared meanRX variance-vector norm; denominator differentiable, floor1e-30',centering='optional perTX/day mean over10packets across5RX, no class or domain fitted head',extra_mode='eval with parameter derivatives; default classW constant only in auxiliary graph; normal L/U head learning unchanged; restore modes/RNG/W flags',no_U_relations=True,no_extra_head=True)
 c['hypothesis_basis']='Existing actual source180 t1/t2 parameter-gradient variance remains RXdependent after TX/day centering; verify trainability on fixed SIG_TIME. Differs from logits or auxiliary-readout gradient proxies. Independent training mechanism, no PAIR or signature-order loss stacking.'
 c['source_falsification']=dict(final_L_CE_max=.5,feature_difference_min=1e-6,actual_extra_batches=EXTRA_BATCHES,gradient_batches=4*EXTRA_PER_EPOCH,all_candidate_gradient_parameters_nonzero_at=[17,33,50],required_gradient_parameters=list(FISHR_PARAMETERS),source_only=True,expansion='Both candidates pass same exact budgets/defaultfeature/source health and actual t1/t2 gradients; expand both fixed02/03 unchanged, no target ranking')
 c['goal']=dict(primary=PRIMARY,clean_three_seed_mean=.84,first_seed_is_not_goal_completion=True,feedback_to_training=False);return c

def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Frozen backbone-gradient variance recipe differs')
def make_args(c,device='cuda:0'):
 validate(c)
 if c['baseline_reuse']:raise ValueError('Do not retrain historical baseline')
 old=p.config(next(r for r in p.training_rows() if r['arm']=='SIG_TIME' and r['model_seed']==c['model_seed']))
 a=p.make_args(old,device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];return a
