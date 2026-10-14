import copy
from pathlib import Path
from experiments.cvs_feature_mean_teacher50 import design as first
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,FULL_FP32_POLICY,read,write=(getattr(first,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','FULL_FP32_POLICY','read','write'))
RUN='20261015-phase1-feature-mean-teacher-oldgate-cos-e50-clean-manysig-s0203-m6-r01';RELEASE='cvs_feature_mean_teacher_seeds_20261015_r02';OWNER='codex/root/feature-mean-teacher-seeds-20261015';BASE=Path(PROJECT)/'runs'/RUN
SEEDS=(2026092702,2026092703)
for name in ('ARMS','BASELINE_ARM','PRIMARY','PAIRS','READOUTS','VIEWS','QUERY_COUNT','EPOCHS','LABEL_EPOCHS','PSEUDO_EPOCHS','STEPS','UPDATES','BASELINE_EVAL_RUN','STYLE_LAYERS','FEATURE_CONTROL','FEATURE_PARAMETER_COUNT','VERIFY_INITIAL_EVAL_EQUAL','PARAMETERS','DIAGNOSTIC_EPOCHS','COEFFICIENT'):globals()[name]=getattr(first,name)
baseline_config=first.baseline_config
def rows():return [dict(row_id=f'{a}-s{s}',arm=a,model_seed=s) for s in SEEDS for a in ARMS]
def training_rows():return [r for r in rows() if r['arm']!=BASELINE_ARM]
def config(row):
 if row not in rows():raise ValueError('Unregistered remaining feature consistency row')
 skeleton=row if row['arm']==BASELINE_ARM else dict(row,arm='SIG_TIME',row_id=f"SIG_TIME-s{row['model_seed']}")
 c=first.p.p.p.parent.configuration(skeleton,run=RUN,base=BASE,owner=OWNER);template=first.config(next(r for r in first.rows() if r['arm']==row['arm']))
 for k in ('feature_combination','source_neighbor_sampling','sampling_observation','feature_mean_teacher','hypothesis_basis','source_falsification','goal'):c[k]=copy.deepcopy(template[k])
 c.update(**row,output_root=(BASE/row['row_id']/'source').as_posix(),source_qualification_run=first.RUN,first_seed_retrained=False,combined_model_seeds=[2026092701,2026092702,2026092703]);return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Remaining fixed feature consistency recipe differs')
def make_args(c,device='cuda:0'):
 validate(c)
 if c['baseline_reuse']:raise ValueError('Historical baseline cannot be retrained')
 a=first.p.p.p.parent.first.old.make_args(baseline_config(c),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];return a
