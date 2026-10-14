"""Report route4: old trusted mask, bounded stopgrad difficulty, actual gradients."""
from pathlib import Path
from experiments.cvs_moment_area50 import design as parent
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,FULL_FP32_POLICY,read,write=(getattr(parent,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','FULL_FP32_POLICY','read','write'))
RUN='20261014-phase1-trusted-weight-oldgate-cos-e50-clean-manysig-s01-m3-r01';RELEASE='cvs_trusted_weight50_20261014_r01';OWNER='codex/root/trusted-weight50-20261014';BASE=Path(PROJECT)/'runs'/RUN
SEEDS=(2026092701,);ARMS=('OLD_GATE_COS50','SIG_TIME','TRUSTED_WEIGHT');BASELINE_ARM=ARMS[0];PRIMARY=ARMS[-1];FEATURE_CONTROL='SIG_TIME';FEATURE_PARAMETER_COUNT=195041;VERIFY_INITIAL_EVAL_EQUAL=True
PAIRS=[('SIG_TIME',BASELINE_ARM),(PRIMARY,'SIG_TIME'),(PRIMARY,BASELINE_ARM)]
READOUTS,VIEWS,QUERY_COUNT,EPOCHS,LABEL_EPOCHS,PSEUDO_EPOCHS,STEPS,UPDATES,BASELINE_EVAL_RUN,STYLE_LAYERS=(getattr(parent,k) for k in ('READOUTS','VIEWS','QUERY_COUNT','EPOCHS','LABEL_EPOCHS','PSEUDO_EPOCHS','STEPS','UPDATES','BASELINE_EVAL_RUN','STYLE_LAYERS'))
def rows():return [dict(row_id=f'{arm}-s{seed}',arm=arm,model_seed=seed) for seed in SEEDS for arm in ARMS]
def training_rows():return [r for r in rows() if r['arm']!=BASELINE_ARM]
baseline_config=parent.baseline_config
def config(row):
 if row not in rows():raise ValueError('Unregistered trusted weight row')
 baseline=row['arm']==BASELINE_ARM;skeleton=row if baseline else dict(row,arm='SIG_TIME',row_id=f"SIG_TIME-s{row['model_seed']}")
 c=parent.configuration(skeleton,run=RUN,base=BASE,owner=OWNER)
 c.update(**row,output_root=(BASE/row['row_id']/'source').as_posix(),trusted_weight=dict(active=row['arm']==PRIMARY,old_gate_unchanged=True,hardness='1-studentstrong probability of teacher pseudo class',raw_weight='stopgrad(.75+.5*clamp(h/.1,0,1))',raw_bounds=[.75,1.25],hardness_scale=.1,normalization='mean over original accepted weight mass',lambda_u_unchanged=.16,entropy_unchanged=True,gradient_epochs=[33,34,50],gradient_batches='all222 actual steps per declared epoch; no new forward or RNG draw',parameter_objects=['t1.pw.weight','t2.pw.weight','t3.pw.weight'],no_U_truth=True,default_joint_head=True),hypothesis_basis='Supplied report route4 and fixed L6300 source-weight observer: accepted difficulty weighting changes actual t1/t2/t3 learning signal. No hierarchy or read-position stacking; no target feedback.',source_falsification=dict(final_L_CE_max=.5,source_added_feature_difference_min=1e-6,actual_gradient_records_required=1998,actual_gradient_epochs=[33,34,50],source_only=True,description='Both finalLCE<=.5, actual feature export/update/W independence and identicalinitialfunction, candidate-control L720 zL2>=1e-6; actual3epochs666batch1998layerrecords/noUtruth/noextraheadfits'),goal=dict(primary=PRIMARY,clean_three_seed_mean=.84,first_seed_is_not_goal_completion=True,feedback_to_training=False))
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Frozen trusted weighting config differs')
def make_args(c,device='cuda:0'):
 validate(c)
 if c['baseline_reuse']:raise ValueError('Historical baseline cannot be retrained')
 a=parent.first.old.make_args(baseline_config(c),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];return a
