"""One Sinc coordinate factor; original feature architecture and default head."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_feature_multiscale50 import design as old
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,SEEDS,FULL_FP32_POLICY,read,write=(getattr(old,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','SEEDS','FULL_FP32_POLICY','read','write'))
RUN='20261014-phase1-sinc-coordinate-oldgate-cos-e50-clean-manysig-m9-r01'
RELEASE='cvs_sinc_coordinates_20261014_r01';OWNER='codex/root/sinc-coordinates-20261014'
BASE=Path(PROJECT)/'runs'/RUN
ARMS=('OLD_GATE_COS50','RAW_HZ','CENTER_25K');BASELINE_ARM='OLD_GATE_COS50';PRIMARY='CENTER_25K'
PAIRS=[('RAW_HZ',BASELINE_ARM),('CENTER_25K',BASELINE_ARM),('CENTER_25K','RAW_HZ')]
READOUTS=('ORIGINAL_HEAD',);VIEWS=('clean',);QUERY_COUNT=168000
SOURCE_DOMAINS=15;EPOCHS=50;LABEL_EPOCHS=33;PSEUDO_EPOCHS=17;STEPS=222;UPDATES=11100
BASELINE_RUN=old.RUN;BASELINE_EVAL_RUN=old.RUN
def rows():return [dict(row_id=f'{a}-s{s}',arm=a,model_seed=s) for s in SEEDS for a in ARMS]
def training_rows():return [r for r in rows() if r['arm']!=BASELINE_ARM]
def baseline_config(row):return old.config(dict(row_id=f"{BASELINE_ARM}-s{row['model_seed']}",arm=BASELINE_ARM,model_seed=row['model_seed']))
def config(row):
 if row not in rows():raise ValueError('Unregistered Sinc coordinate row')
 prior=baseline_config(row);c=deepcopy(prior);arm=row['arm']
 c.update(schema='sinc_coordinate_default_v1',run_id=RUN,**row,method='cvs_sinc_coordinates',stage='evaluation_reuse' if arm==BASELINE_ARM else 'sinc_coordinate',output_root=(BASE/row['row_id']/'source').as_posix(),launch_owner=OWNER,source_config=prior if arm==BASELINE_ARM else None,model_initialization='historical comparison only' if arm==BASELINE_ARM else 'scratch',
  sinc_coordinates=dict(enabled=arm!=BASELINE_ARM,mode=arm if arm!=BASELINE_ARM else 'native',scale_hz=25000.,insertion='actual identity Sinc before native time_down',trainable_parameters=48,extra_trainable_parameters=0,origin_buffer_elements=48,initial_native_filters_exact=True,initial_whole_joint160_exact=True,initial_function_same=True,init_RNG_preserved=True,unchanged=['native_time','frequency','PA','reference','input_policy','default_head','old_SSL']),
  readouts=list(READOUTS),classifier_training='default jointly trained only; no separate head fit',
  source_falsification=dict(first_seed=SEEDS[0],maximum_final_L_CE=.5,minimum_center_final_kernel_relative_L2=1e-4,metric='fixedfinal source LCE health and actual initial-to-final Sinc kernel movement; no V or target ranking'),
  goal=dict(primary=PRIMARY,readout='ORIGINAL_HEAD',clean_three_seed_mean=.84,feedback_to_training=False),
  hypothesis_basis='Actual9source models:11100 Sinc optimizer states,47 nonzero moments,near-static student filters,final adaptive terms belowhalfULP;stationary EMA reproduces saved teacher kernels',
  no_TIME_variant_inheritance=True,classifier='unchanged ResidualPhysicalHead cosine6x160scale30 normal joint training',
  coordinate_scope='Combined centered+scaled parameterization, including changed optimizer/weight-decay/EMA numerical coordinates; not a pure causal test of scaling alone')
 c.pop('feature_extractor',None)
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row) or c['epochs']>50 or c['readouts']!=['ORIGINAL_HEAD']:raise ValueError('Fixed Sinc/default-head matrix differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c)
 if c['arm']==BASELINE_ARM:raise ValueError('Historical baseline cannot be retrained')
 a=old.make_args(baseline_config(c),device);a.output_dir=c['output_root'];a.run_id=RUN;a.candidate_id=c['row_id'];a.seed=c['model_seed']
 return a
