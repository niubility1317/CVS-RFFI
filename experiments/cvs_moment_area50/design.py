"""One-factor same-projection statistical-shape versus ordered-area experiment."""
from pathlib import Path
from experiments.cvs_signature_stats50 import design as first
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,FULL_FP32_POLICY,read,write=(getattr(first,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','FULL_FP32_POLICY','read','write'))
RUN='20261014-phase1-moment-area-oldgate-cos-e50-clean-manysig-s01-m4-r01'
RELEASE='cvs_moment_area50_20261014_r01';OWNER='codex/root/moment-area-20261014';BASE=Path(PROJECT)/'runs'/RUN
SEEDS=(2026092701,);ARMS=('OLD_GATE_COS50','SIG_TIME','COV_SHAPE','AREA_SHAPE');BASELINE_ARM=ARMS[0];PRIMARY=ARMS[-1]
FEATURE_CONTROL='COV_SHAPE';FEATURE_PARAMETER_COUNT=195041
PAIRS=[('SIG_TIME',BASELINE_ARM),('COV_SHAPE','SIG_TIME'),(PRIMARY,FEATURE_CONTROL),(PRIMARY,'SIG_TIME'),(PRIMARY,BASELINE_ARM)]
READOUTS,VIEWS,QUERY_COUNT,EPOCHS,LABEL_EPOCHS,PSEUDO_EPOCHS,STEPS,UPDATES,BASELINE_EVAL_RUN,STYLE_LAYERS=(getattr(first,k) for k in ('READOUTS','VIEWS','QUERY_COUNT','EPOCHS','LABEL_EPOCHS','PSEUDO_EPOCHS','STEPS','UPDATES','BASELINE_EVAL_RUN','STYLE_LAYERS'))
def rows():return [dict(row_id=f'{arm}-s{seed}',arm=arm,model_seed=seed) for seed in SEEDS for arm in ARMS]
def training_rows():return [r for r in rows() if r['arm']!=BASELINE_ARM]
baseline_config=first.baseline_config
def config(row):
 if row not in rows():raise ValueError('Unregistered moment-area row')
 return configuration(row,run=RUN,base=BASE,owner=OWNER)
def configuration(row,*,run,base,owner):
 baseline=row['arm']==BASELINE_ARM
 skeleton=row if baseline else dict(row,arm='SIG_TIME',row_id=f"SIG_TIME-s{row['model_seed']}")
 c=first.configuration(skeleton,run=run,base=base,owner=owner)
 c.update(**row,output_root=(base/row['row_id']/'source').as_posix(),feature_combination=dict(modes=[] if baseline else [row['arm']],layer='t2',local_dimension=6,windows=4,descriptor_width_each=28,descriptor_raw_width=28 if row['arm']=='SIG_TIME' else 27,fixed_lift='SIG_TIME identity28; COV_SHAPE/AREA_SHAPE same original COV orthonormal27to28 seed2026101307',shared_single_projection=True,gain_initial=.05,gain_bound=.2,output_embedding=160),hypothesis_basis='User supplied report section3 route1; preserve mean6/variance6 and replace15 offdiagonal covariance slots with skew area, same six-dimensional path. No descriptor/Dropout/hierarchy/U-weight stacking and no target feedback.',source_falsification=dict(final_L_CE_max=.5,source_added_feature_difference_min=1e-6,source_only=True,description='All3 finalLCE<=.5, real joint160 W-independent export, actual new parameter updates and AREA vs COV sourceL720 difference>=1e-6; predefined02/03 only from source verdict'),goal=dict(primary=PRIMARY,clean_three_seed_mean=.84,first_seed_is_not_goal_completion=True,feedback_to_training=False))
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Frozen moment-area configuration differs')
def make_args(c,device='cuda:0'):
 validate(c)
 if c['baseline_reuse']:raise ValueError('Historical baseline cannot be retrained')
 a=first.old.make_args(baseline_config(c),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];return a
