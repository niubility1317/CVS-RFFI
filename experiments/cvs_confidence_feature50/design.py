from copy import deepcopy
from pathlib import Path
from experiments.cvs_sigtime_pair50 import design as p
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,FULL_FP32_POLICY,read,write=(getattr(p,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','FULL_FP32_POLICY','read','write'))
RUN="20261015-phase1-confidence-feature-oldgate-cos-e50-clean-manysig-s01-m3-r01";RELEASE="cvs_confidence_feature50_20261015_r01";OWNER="codex/root/confidence-feature50-20261015";BASE=Path(PROJECT)/'runs'/RUN
SEEDS=(2026092701,);ARMS=('OLD_GATE_COS50','SIG_TIME','CONF_FEATURE_MT');BASELINE_ARM=ARMS[0];PRIMARY=ARMS[-1];FEATURE_CONTROL='SIG_TIME';FEATURE_PARAMETER_COUNT=195041;VERIFY_INITIAL_EVAL_EQUAL=True
PAIRS=[(PRIMARY,FEATURE_CONTROL),(PRIMARY,BASELINE_ARM),(FEATURE_CONTROL,BASELINE_ARM)]
READOUTS,VIEWS,QUERY_COUNT,EPOCHS,LABEL_EPOCHS,PSEUDO_EPOCHS,STEPS,UPDATES,BASELINE_EVAL_RUN,STYLE_LAYERS,PARAMETERS,DIAGNOSTIC_EPOCHS=(getattr(p,k) for k in ('READOUTS','VIEWS','QUERY_COUNT','EPOCHS','LABEL_EPOCHS','PSEUDO_EPOCHS','STEPS','UPDATES','BASELINE_EVAL_RUN','STYLE_LAYERS','PARAMETERS','DIAGNOSTIC_EPOCHS'))
baseline_config=p.baseline_config
COEFFICIENT=.1
def rows():return [dict(row_id=f'{a}-s{s}',arm=a,model_seed=s) for s in SEEDS for a in ARMS]
def training_rows():return [r for r in rows() if r['arm']!=BASELINE_ARM]
def config(row):
 if row not in rows():raise ValueError('Unregistered feature mean-teacher row')
 skeleton=row if row['arm']==BASELINE_ARM else dict(row,arm='SIG_TIME',row_id=f"SIG_TIME-s{row['model_seed']}")
 c=deepcopy(p.config(skeleton));c.update(run_id=RUN,**row,output_root=(BASE/row['row_id']/'source').as_posix(),launch_owner=OWNER)
 c['feature_mean_teacher']=dict(mode='ZERO' if row['arm']==FEATURE_CONTROL else 'FEATURE' if row['arm']==PRIMARY else 'historical',feature='actual deployed id_feat_joint160',teacher='actual own EMA weak view, stopgradient; unchanged .999 updates',student='existing strong Gaussian.015 and original trainmode dropout',loss='Fixed detached FP32 ownEMA weak maxprob >= student strong maxprob, ties kept; .5 unitjoint160 squaredL2 times gate summed/full batchN eps1e-8; no RX/day/TX truth in objective',coefficient=COEFFICIENT,pseudo_start_epoch=34,ramp_epochs=5,extra_model_forwards=0,new_modules=0,new_parameters=0,new_heads=0,hard_pseudo_gate='original batch_neighbor and all original losses unchanged',diagnostic_epochs=list(DIAGNOSTIC_EPOCHS),parameters=list(PARAMETERS))
 c['hypothesis_basis']='Fixed six-model source-only confidence-order qualification PASS 20261015-phase1-source-teacher-confidence-manysig-m6-r01; sameL helpful/harmful feature-gradient directions differ. Test single confidence-ordered teacher feature imitation, no target-derived parameters or Fishr stacking.'
 c['source_falsification']=dict(final_L_CE_max=.5,source_feature_difference_min=1e-6,actual_feature_gradients_nonzero_at=[34,50],last5_retained_feature_mean_ratio_max=.95,last5_allU_feature_mean_ratio_max=1.,last5_candidate_retention_min=.5,source_only=True,expansion='Source health/exact11100 updates/all17x56700 U/featureboundary/nonzero actual t1t2SIG plus retained mean ratio<=.95,allU discrepancy ratio<=1.,coverage>=.5. Freeze decision before any new target summary retrieval; predefined02/03 only and no target ranking, parameter change or selective rerun.')
 c['confidence_gate']=dict(rule='FP32 teacherweak maxsoftmax >= studentstrong maxsoftmax',gap_threshold=0.,ties_kept=True,detached=True,denominator='full batch N',source_qualification_run='20261015-phase1-source-teacher-confidence-manysig-m6-r01',original_U_CE_and_entropy='unchanged',extra_model_forwards=0,new_parameters=0,new_heads=0)
 c['goal']=dict(primary=PRIMARY,clean_three_seed_mean=.84,first_seed_is_not_goal_completion=True,feedback_to_training=False);return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Frozen feature consistency recipe differs')
def make_args(c,device='cuda:0'):
 validate(c)
 if c['baseline_reuse']:raise ValueError('Do not retrain historical baseline')
 old=p.config(next(r for r in p.training_rows() if r['arm']=='SIG_TIME' and r['model_seed']==c['model_seed']))
 a=p.make_args(old,device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];return a
