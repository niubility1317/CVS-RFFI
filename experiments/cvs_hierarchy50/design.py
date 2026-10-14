"""One-factor same-projection statistical-shape versus ordered-area experiment."""
from pathlib import Path
from experiments.cvs_signature_stats50 import design as first
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,FULL_FP32_POLICY,read,write=(getattr(first,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','FULL_FP32_POLICY','read','write'))
RUN='20261014-phase1-hierarchy-oldgate-cos-e50-clean-manysig-s01-m5-r01'
RELEASE='cvs_hierarchy50_20261014_r01';OWNER='codex/root/hierarchy-20261014';BASE=Path(PROJECT)/'runs'/RUN
SEEDS=(2026092701,);ARMS=('OLD_GATE_COS50','SIG_TIME','LOCAL_GTIME','HIER_STAT','HIER_ORDER');BASELINE_ARM=ARMS[0];PRIMARY=ARMS[-1]
FEATURE_CONTROL='HIER_STAT';FEATURE_PARAMETER_COUNTS={'SIG_TIME':195041,'LOCAL_GTIME':195041,'HIER_STAT':208481,'HIER_ORDER':208481}
PAIRS=[('SIG_TIME',BASELINE_ARM),('LOCAL_GTIME','SIG_TIME'),('HIER_STAT','LOCAL_GTIME'),(PRIMARY,FEATURE_CONTROL),(PRIMARY,'LOCAL_GTIME'),(PRIMARY,'SIG_TIME'),(PRIMARY,BASELINE_ARM)]
READOUTS,VIEWS,QUERY_COUNT,EPOCHS,LABEL_EPOCHS,PSEUDO_EPOCHS,STEPS,UPDATES,BASELINE_EVAL_RUN,STYLE_LAYERS=(getattr(first,k) for k in ('READOUTS','VIEWS','QUERY_COUNT','EPOCHS','LABEL_EPOCHS','PSEUDO_EPOCHS','STEPS','UPDATES','BASELINE_EVAL_RUN','STYLE_LAYERS'))
def rows():return [dict(row_id=f'{arm}-s{seed}',arm=arm,model_seed=seed) for seed in SEEDS for arm in ARMS]
def training_rows():return [r for r in rows() if r['arm']!=BASELINE_ARM]
baseline_config=first.baseline_config
def config(row):
 if row not in rows():raise ValueError('Unregistered hierarchy row')
 return configuration(row,run=RUN,base=BASE,owner=OWNER)
def configuration(row,*,run,base,owner):
 baseline=row['arm']==BASELINE_ARM
 skeleton=row if baseline else dict(row,arm='SIG_TIME',row_id=f"SIG_TIME-s{row['model_seed']}")
 c=first.configuration(skeleton,run=run,base=base,owner=owner)
 c.update(**row,output_root=(base/row['row_id']/'source').as_posix(),feature_combination=dict(modes=[] if baseline else [row['arm']],layer='t2',local_dimension=6,windows=4,hierarchy=[4,2,1] if row['arm'] in ('HIER_STAT','HIER_ORDER') else [4],descriptor_width_each=28,total_descriptor_width=196 if row['arm'] in ('HIER_STAT','HIER_ORDER') else 112,time_coordinate='localreset0to1' if row['arm']=='SIG_TIME' else 'one global32 linspace0to1',three_boundary_increments=row['arm']=='HIER_ORDER',coarse_context='mean6/variance6/cov15 fixed27to28lift' if row['arm']=='HIER_STAT' else 'continuous increment7/skewarea21' if row['arm']=='HIER_ORDER' else 'none',shared_single_projection=True,main_body_dropout_unchanged=True,gain_initial=.05,gain_bound=.2,output_embedding=160),hypothesis_basis='User report route2 plus fixed source L720 continuous-signature geometry; add genuine adjacent16/full32 continuous descriptors with3bridges/globalclock. Compare originalSIG_TIME, global-clock-only local control and matched-capacity large-window statistics. No preDropout/Uweight/kernel/Sinc changes; no target-based checkpoint selection.',source_diagnostic_run='20261014-phase1-source-hierarchy-geometry-manysig-s01-m1-r01',source_falsification=dict(final_L_CE_max=.5,source_added_feature_difference_min=1e-6,source_only=True,description='All4 finalLCE<=.5; actual addedfeature numeric updates/exact export/W independence; matchedHIER_STAT/HIER_ORDER initial trainable parameters and nativebody/W equal, final L720 HIER_ORDER-HIER_STAT L2>=1e-6; unchanged02/03 only from sourcePASS, no targetmetric'),goal=dict(primary=PRIMARY,clean_three_seed_mean=.84,first_seed_is_not_goal_completion=True,feedback_to_training=False))
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Frozen hierarchy configuration differs')
def make_args(c,device='cuda:0'):
 validate(c)
 if c['baseline_reuse']:raise ValueError('Historical baseline cannot be retrained')
 a=first.old.make_args(baseline_config(c),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];return a
