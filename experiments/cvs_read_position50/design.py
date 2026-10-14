"""One-factor SIG_TIME descriptor read before vs after native t2 Dropout."""
from pathlib import Path
from experiments.cvs_signature_stats50 import design as first
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,FULL_FP32_POLICY,read,write=(getattr(first,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','FULL_FP32_POLICY','read','write'))
RUN='20261014-phase1-read-position-oldgate-cos-e50-clean-manysig-s01-m3-r01'
RELEASE='cvs_read_position50_20261014_r01';OWNER='codex/root/read-position-20261014';BASE=Path(PROJECT)/'runs'/RUN
SEEDS=(2026092701,);ARMS=('OLD_GATE_COS50','SIG_TIME','SIG_TIME_PRE');BASELINE_ARM=ARMS[0];PRIMARY=ARMS[-1]
FEATURE_CONTROL='SIG_TIME';FEATURE_PARAMETER_COUNT=195041;VERIFY_INITIAL_EVAL_EQUAL=True
PAIRS=[('SIG_TIME',BASELINE_ARM),(PRIMARY,FEATURE_CONTROL),(PRIMARY,BASELINE_ARM)]
READOUTS,VIEWS,QUERY_COUNT,EPOCHS,LABEL_EPOCHS,PSEUDO_EPOCHS,STEPS,UPDATES,BASELINE_EVAL_RUN,STYLE_LAYERS=(getattr(first,k) for k in ('READOUTS','VIEWS','QUERY_COUNT','EPOCHS','LABEL_EPOCHS','PSEUDO_EPOCHS','STEPS','UPDATES','BASELINE_EVAL_RUN','STYLE_LAYERS'))
def rows():return [dict(row_id=f'{arm}-s{seed}',arm=arm,model_seed=seed) for seed in SEEDS for arm in ARMS]
def training_rows():return [r for r in rows() if r['arm']!=BASELINE_ARM]
baseline_config=first.baseline_config
def config(row):
 if row not in rows():raise ValueError('Unregistered read-position row')
 return configuration(row,run=RUN,base=BASE,owner=OWNER)
def configuration(row,*,run,base,owner):
 baseline=row['arm']==BASELINE_ARM
 skeleton=row if baseline else dict(row,arm='SIG_TIME',row_id=f"SIG_TIME-s{row['model_seed']}")
 c=first.configuration(skeleton,run=run,base=base,owner=owner)
 c.update(**row,output_root=(base/row['row_id']/'source').as_posix(),feature_combination=dict(modes=[] if baseline else ['SIG_TIME'],layer='t2.pool' if row['arm']=='SIG_TIME_PRE' else 't2',local_dimension=6,windows=4,descriptor_width_each=28,output_embedding=160,gain_initial=.05,gain_bound=.2,output_parameter_matched=True,main_body_dropout_unchanged=True),hypothesis_basis='User report section3 route3 plus frozen-source Dropout observer: feature area noise/cleanRMS31.9940% vs time21.7255% on L720. Compare read position only; no descriptor/capacity/hierarchy/Dropout probability/U weighting changes or target-based checkpoint selection.',source_diagnostic_run='20261014-phase1-source-dropout-descriptor-manysig-s01-m1-r01',source_falsification=dict(final_L_CE_max=.5,source_added_feature_difference_min=1e-6,source_only=True,description='Both2 finalLCE<=.5; actual newfeature updates/exact export/W independence, same initial full parameters and initial evalfunction; PRE-POST finalL720L2>=1e-6. Only unchanged predefined02/03 from source PASS.'),goal=dict(primary=PRIMARY,clean_three_seed_mean=.84,first_seed_is_not_goal_completion=True,feedback_to_training=False))
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Frozen read-position configuration differs')
def make_args(c,device='cuda:0'):
 validate(c)
 if c['baseline_reuse']:raise ValueError('Historical baseline cannot be retrained')
 a=first.old.make_args(baseline_config(c),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];return a
