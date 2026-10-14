"""User-selected SIG_TIME statistics combination; fixed firstseed and controls."""
from pathlib import Path
from experiments.cvs_feature_multiscale50 import design as old
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,FULL_FP32_POLICY,read,write=(getattr(old,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','FULL_FP32_POLICY','read','write'))
RUN='20261014-phase1-sigtime-statistics-oldgate-cos-e50-clean-manysig-s01-m4-r01'
RELEASE='cvs_signature_stats50_20261014_r01';OWNER='codex/root/signature-statistics-20261014';BASE=Path(PROJECT)/'runs'/RUN
SEEDS=(2026092701,);ARMS=('OLD_GATE_COS50','SIG_TIME','SIG_TIME_X3','SIG_TIME_MV_COV');BASELINE_ARM=ARMS[0];PRIMARY=ARMS[-1]
PAIRS=[('SIG_TIME',BASELINE_ARM),('SIG_TIME_X3','SIG_TIME'),(PRIMARY,'SIG_TIME'),(PRIMARY,'SIG_TIME_X3'),(PRIMARY,BASELINE_ARM)]
READOUTS=('ORIGINAL_HEAD',);VIEWS=('clean',);QUERY_COUNT=168000
EPOCHS=50;LABEL_EPOCHS=33;PSEUDO_EPOCHS=17;STEPS=222;UPDATES=11100
BASELINE_EVAL_RUN='20261013-phase1-feature-multiscale-oldgate-cos-e50-clean-manysig-m12-r01'
STYLE_LAYERS=()
def rows():return [dict(row_id=f'{arm}-s{seed}',arm=arm,model_seed=seed) for seed in SEEDS for arm in ARMS]
def training_rows():return [r for r in rows() if r['arm']!=BASELINE_ARM]
def baseline_config(row):return old.config(next(r for r in old.rows() if r['arm']==BASELINE_ARM and r['model_seed']==row['model_seed']))
def config(row):
 if row not in rows():raise ValueError('Unregistered signature statistics row')
 return configuration(row,run=RUN,base=BASE,owner=OWNER)
def configuration(row,*,run,base,owner):
 ref=baseline_config(row);baseline=row['arm']==BASELINE_ARM
 return dict(ref,schema='signature_stats_default50_v1',run_id=run,**row,launch_owner=owner,output_root=(base/row['row_id']/'source').as_posix(),source_config=ref if baseline else None,baseline_reuse=baseline,model_initialization='historical comparison only' if baseline else 'scratch',feature_combination=dict(modes=[] if baseline else list({'SIG_TIME':('SIG_TIME',),'SIG_TIME_X3':('SIG_TIME','SIG_TIME','SIG_TIME'),'SIG_TIME_MV_COV':('SIG_TIME','MEAN_VAR','COV')}[row['arm']]),layer='t2',independent_projection_each=True,local_dimension=6,windows=4,descriptor_width_each=28,gain_initial=.05,gain_bound=.2,output_embedding=160,parameter_matched_to_X3=row['arm']==PRIMARY),classifier='unchanged default cosine6x160scale30 normal joint training only',readouts=list(READOUTS),test_views=list(VIEWS),separate_head_training=False,training_recipe='old batch_neighbor gate+EMA+cos scratch50 label33+pseudo17, no satellite',source_falsification=dict(final_L_CE_max=.5,source_added_feature_difference_min=1e-6,source_only=True,description='All3 new final LCE<=.5, real default feature export exact/W-independent, added modules numerically updated and candidate differs from same-capacity X3 on fixedL720. Future predefined seed expansion only from these source facts.'),hypothesis_basis='User explicitly selects SIG_TIME as base on exposed benchmark; add independently learned moment/covariance views and compare same-parameter3SIG_TIME; no target feedback to parameters or checkpoint/seed choice',source_formula_parent_commit='990ffabe0',initialization_policy='Warm CUDA before native training seed; private pool RNG preserves baseline draws; verify actual initial native body/head equality',goal=dict(primary=PRIMARY,clean_three_seed_mean=.84,first_seed_is_not_goal_completion=True,feedback_to_training=False))
def validate(c):
 if c!=config(next((r for r in rows() if r['row_id']==c.get('row_id')),{})):raise ValueError('Frozen signature statistics config differs')
def make_args(c,device='cuda:0'):
 validate(c);a=old.make_args(baseline_config(c),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];return a
