"""User-directed full-source E20/E50 three-seed clean comparison; no source gate."""
from pathlib import Path
from experiments.cvs_sigtime_rdrop_legacy import design as native
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,SEEDS,FULL_FP32_POLICY,read,write=(getattr(native,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','SEEDS','FULL_FP32_POLICY','read','write'))
RUN='20261016-phase1-SIGTIME-rdrop-feature-mechanisms-e20e50-clean-manysig-m12-r01'
RELEASE='cvs_SIGTIME_rdrop_feature_mechanisms_main20_50_20261016_r01'
OWNER='codex/root/SIGTIME-RDrop-features-direct-test-20261016'
BASE=Path(PROJECT)/'runs'/RUN;BUDGETS=(20,50);MODES=('POINT','DYNAMIC');ARMS=tuple(f'E{e}_{m}' for e in BUDGETS for m in MODES)
VIEWS=('clean',);READOUTS=('NATIVE_HEAD',)
MODEL_RELEASE=PROJECT+'/releases/cvs_originalSIGTIME_rdrop20_50_20261015_r01';MODEL_COMMIT='b3ac442e2d33a5b315fb3251c30e0ad7083566bc'
GAIN_RELEASE=PROJECT+'/releases/cvs_SIGTIME_rdrop_gradient_gain_RX1_source20_20261016_r01';GAIN_COMMIT='9e3034698962f6a5c61b37f92f63a80a1a82ae03'
CENTER_RELEASE=PROJECT+'/releases/cvs_SIGTIME_rdrop_batch_center_RX3_source20_20261016_r01';CENTER_COMMIT='5c60deca3f42c02ebf82f5402a205bd9db81c89f'
DEPENDENCY_RELEASE,DEPENDENCY_COMMIT,FOLD_RELEASE=(getattr(native,k) for k in ('DEPENDENCY_RELEASE','DEPENDENCY_COMMIT','FOLD_RELEASE'))
HISTORY_ORIGINAL=native.RUN;HISTORY_RECOVERY='20261015-phase1-originalSIGTIME-rdrop-e50-technical-clean-manysig-m4-r02'
stages=native.stages
def rows():return [dict(row_id=f'E{e}_{m}-s{s}',arm=f'E{e}_{m}',mode=m,epochs=e,model_seed=s) for s in SEEDS for e in BUDGETS for m in MODES]
def native_config(c):return native.config(next(r for r in native.rows() if r['epochs']==c['epochs'] and r['model_seed']==c['model_seed'] and r['mode']=='RDROP'))
def config(row):
 if row not in rows():raise ValueError('Fixed12 direct-test feature rows required')
 c=native_config(row)
 return dict(c,schema='SIGTIME_RDrop_feature_mechanisms_direct_test_v1',run_id=RUN,**row,updates=row['epochs']*222,launch_owner=OWNER,
  output_root=(BASE/row['row_id']/'source').as_posix(),baseline_reuse=False,baseline_reference='Six exact matched original RDROP scored artifacts reused;no source-fold or historical trained weights loaded',source_falsification=None,
  cost_claim='12 new scratch models:420 model-epoch/93240 optimizer updates;6 reused scored RDROP controls:210 historical model-epoch;no head refit',
  source_selection=None,training_recipe=f'Original SIG_TIME RAW_HZ/normalaug/LS.01/defaultcos joint;full-five-source scratch{row["epochs"]} '+str(stages(row['epochs']))+' L/U;native fullKL plus '+row['mode'],
  training_mechanism=dict(mode=row['mode'],alpha=1.,label_smoothing=.01,
   objective='Original paired smoothed CE + full symmetric KL + selected uncentered feature consistency + original hiddenU',
   P='.5*mean(stop(T)*squared(normalize(z1)-normalize(z2))/160)',
   C_reference='.5*mean(stop(T)*squared(normalize(z1-mu)-normalize(z2-mu))/160);mu=.5*mean(z1+z2),differentiable',
   T='900*(sum_k pbar_k*norm(normalize(W)_k)^2-norm(pbar@normalize(W))^2),pbar=mean softmax(logits1,logits2),detached,double calculation',
   applied_gain='1' if row['mode']=='POINT' else 'stop(norm(gC)/norm(gP)) at concatenated actual joint160 pair,unchanged released operator',
   normalization_eps=1e-4,ratio_clipping=False,normalization_reference_only=True,default_W_joint=True,auxiliary_default_W_direct_gradient=False,
   original_U_sampler_unchanged=True,augmentation_redraws=0,extra_parameters=0,extra_optimizer_steps=0,extra_L_identity_forwards=row['epochs']*222*128,
   boundary_autograd_traversals_per_step=2,scope='Training-only feature loss;no inference transform/state,head refit or frozen backbone'),
  budget_authorization='User explicitly directs actual test performance instead of held-source receivers;retain E20/E50 three fixed seeds. No held-RX/source-health scientific release gate,12new scratch+6reused scored controls.',
  goal=dict(primary=['E50_POINT','E50_DYNAMIC'],clean_three_seed_mean=.84,mean_gain_over_same_budget_RDROP_strictly_positive=True,positive_seeds_min=2,E20_budget_comparison=True,
   source_scores_not_release_gate=True,all_registered_rows_test_automatically=True,global_novelty_claim=False,target_feedback=False,goal_completed=False))
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed full-source direct-test matrix differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c);a=native.make_args(native_config(c),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root']
 if getattr(a,'source_fold_held_rx',None) is not None or a.baseline_ckpt or a.teacher_ckpt:raise ValueError('Full original source and own scratch required')
 return a
def historical_rows():
 result=[]
 for s in SEEDS:
  for e in BUDGETS:
   rid=f'E{e}_RDROP-s{s}';run=HISTORY_RECOVERY if e==50 and s in SEEDS[1:] else HISTORY_ORIGINAL
   result.append(dict(row_id=rid,arm=f'E{e}_RDROP',mode='RDROP',epochs=e,model_seed=s,run_id=run,root=(Path(PROJECT)/'runs'/run/rid).as_posix()))
 return result
def totals():return dict(new_models=12,new_epochs=420,new_optimizer_steps=93240,extra_L_identity_forwards=11934720,U_observations=2*3*(7+17)*56700,new_clean_decisions=2016000,new_metric_records=216,reused_models=6,reused_epochs=210,reused_clean_decisions=1008000,compared_rows=18)
def request():return dict(run_id=RUN,model_release=MODEL_RELEASE,model_commit=MODEL_COMMIT,gain_release=GAIN_RELEASE,gain_commit=GAIN_COMMIT,center_release=CENTER_RELEASE,center_commit=CENTER_COMMIT,dependency_release=DEPENDENCY_RELEASE,fold_release=FOLD_RELEASE,rows=[config(r) for r in rows()],historical_rows=historical_rows(),views=list(VIEWS),readouts=list(READOUTS),default_head_only=True)
