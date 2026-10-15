"""Confirm the fixed feature-gradient gain on a different source receiver fold."""
from pathlib import Path
import math
from experiments.cvs_sigtime_rdrop_legacy import design as original
from experiments.cvs_rx_u_increment20.design import source_validation
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write=(getattr(original,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write'))
RUN='20261016-phase1-SIGTIME-rdrop-gradient-gain-RX1-sourcefold-s2026092702-m4-r01'
RELEASE='cvs_SIGTIME_rdrop_gradient_gain_RX1_source20_20261016_r01'
OWNER='codex/root/SIGTIME-RDrop-gradient-gain-20261016'
MODEL_RELEASE=PROJECT+'/releases/cvs_originalSIGTIME_rdrop20_50_20261015_r01'
MODEL_COMMIT='b3ac442e2d33a5b315fb3251c30e0ad7083566bc'
CENTER_RELEASE=PROJECT+'/releases/cvs_SIGTIME_rdrop_batch_center_RX3_source20_20261016_r01'
CENTER_COMMIT='5c60deca3f42c02ebf82f5402a205bd9db81c89f'
DEPENDENCY_RELEASE,DEPENDENCY_COMMIT,FOLD_RELEASE=(getattr(original,k) for k in ('DEPENDENCY_RELEASE','DEPENDENCY_COMMIT','FOLD_RELEASE'))
BASE=Path(PROJECT)/'runs'/RUN
SEED,FIT,HELD,RX,STEPS,UPDATES=2026092702,(3,4,6,8),1,(1,3,4,6,8),222,4440
ARMS=('FULL','POINT','FIXED11','DYNAMIC')
PREVIOUS_RATIO_SUM=46868.09065139294
FIXED_GAIN=round(PREVIOUS_RATIO_SUM/UPDATES)
assert FIXED_GAIN==11

def native_config():
 return original.config(next(r for r in original.rows() if r['epochs']==20 and r['mode']=='RDROP' and r['model_seed']==SEED))
def rows():return [dict(row_id=a+'-RX1-s'+str(SEED),arm=a,model_seed=SEED) for a in ARMS]
def config(row):
 if row not in rows():raise ValueError('Fixed four RX1 gradient-gain rows required')
 return dict(native_config(),schema='SIGTIME_RDrop_gradient_gain_v1',run_id=RUN,**row,launch_owner=OWNER,
  output_root=(BASE/row['row_id']/'source').as_posix(),method='cvs_rdrop_gradient_gain',stage='source_RX_fold_full_training',test_views=[],readouts=['FINAL'],updates=UPDATES,source_falsification=None,
  fold=dict(held_RX=HELD,training_RX=list(FIT),effective_roles=dict(L_s=5040,U_s=45360,V=21600),held_L_evaluation=1260,
   carrier_source_roles='Original VALIDATED_ONCE L6300/U56700/V27000;metadata verified then all five loaders exclude RX1',
   all_model_teacher_training_calibration_and_feature_cache_inputs_exclude_RX1=True,raw_source_carrier_and_static_domain_ID_map_retained=True,
   trained_checkpoint_reuse_allowed_for_all5RX=False,steps_per_epoch=222,U_observations_per_pseudo_epoch=56624),
  training_recipe='Unchanged original SIG_TIME RAW_HZ/normalaug/LS.01/defaultcos joint;scratch20 13L+7oldU,true RX1 exclusion,222steps',
  training_mechanism=dict(arm=row['arm'],alpha=1.,label_smoothing=.01,
   FULL='Exact original full symmetric KL',POINT='FULL + P',FIXED11='FULL + 11*P',
   DYNAMIC='FULL + stop(norm(g_C)/norm(g_P))*P;exact prior RX3 NORM_MATCH operator',
   P='.5*mean(stop(T)*squared(normalize(z1)-normalize(z2))/160)',
   C='Reference only:.5*mean(stop(T)*squared(normalize(z1-mu)-normalize(z2-mu))/160),mu=.5*mean(z1+z2),differentiable',
   Fisher_trace='T=900*(sum_k pbar_k*norm(normalize(W)_k)^2-norm(pbar@normalize(W))^2);pbar=mean softmax(logits1,logits2),detached,double calculation',
   normalization_eps=1e-4,norm_ratio_stop_gradient=True,reference_mean_stop_gradient=False,
   norm_match_space='Concatenated actual raw joint160 pair only;not model parameters,clipping or AdamW',norm_match_tolerance=2e-5,common_translation_gradient_ratio_tolerance=1e-5,
   fixed_gain=FIXED_GAIN,fixed_gain_origin=dict(source_run='20261016-phase1-SIGTIME-rdrop-batch-center-RX3-sourcefold-s2026092702-m4-r01',
    source_arm='NORM_MATCH',ratio_sum=PREVIOUS_RATIO_SUM,training_calls=UPDATES,rule='Nearest integer of all4440 training-graph ratios;no sweep or held/target-score fitting'),
   objective='Original paired smoothed CE + full native KL + selected uncentered feature auxiliary + original U',
   original_hidden_U_recipe_unchanged=True,default_head_joint=True,KL_default_W_joint=True,auxiliary_default_W_direct_gradient=False,
   augmentation_redraws=0,extra_parameters=0,extra_optimizer_steps=0,extra_L_identity_forwards=568320,
   boundary_autograd_traversals_per_step=0 if row['arm']=='FULL' else 2,
   interpretation='Reference C supplies only a scalar norm;applied auxiliary direction always uncentered P. No encoder-center/RX/optimizer invariance or global novelty guarantee'),
  source_selection=dict(primary='DYNAMIC',controls=['FULL','POINT','FIXED11'],held_RX=HELD,readout='Fixed final20 joint default head,1260 held source L',
   final_L_CE_max=.5,minimum_aux_positive_call_fraction=.9,errors_strictly_fewer=True,mean_CE_no_higher=True,
   on_pass='Supports separately preregistered full-source E20/E50 three-seed original-RDrop comparison with automatic clean truth-last;no target in this source run',
   on_fail='Close fixed gradient-gain candidate;no ratio/weight/cap/seed/epoch/temperature rescue or posthoc substitution of FIXED11',target_access=False),
  baseline_reference='All four own scratch,shared initial seed and old original construction;no trained ancestor;reused dependencies are code only',
  budget_authorization='Latest user only SIG_TIME+RDrop;confirm exploratory prior RX3 NORM_MATCH on RX1 and distinguish dynamic gain from fixed large weight,4scratchE20',
  cost_claim='80model-epoch/17760updates;each568320extraL/396368hiddenUobservations/1260heldL/144unusedfitL anchors;0target',
  goal=dict(clean_three_seed_mean=.84,source_only=True,target_access=False,global_novelty_claim=False,goal_completed=False))
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed gradient-gain source confirmation differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c);a=original.make_args(native_config(),device)
 a.run_id,a.candidate_id,a.output_dir=RUN,c['row_id'],c['output_root'];a.source_fold_held_rx=HELD;a.source_fold_training_receivers=','.join(map(str,FIT))
 if (a.epochs,a.label_epochs,a.pseudo_epochs,a.label_smoothing,a.use_aug,a.batch_size)!=(20,13,7,.01,True,128) or a.use_concat_sat_channel_aug or a.baseline_ckpt or a.teacher_ckpt:raise ValueError('Original source scratch recipe required')
 return a
def qualify(results):
 if set(results)!=set(ARMS):raise ValueError('All four source results required')
 for r in results.values():
  h=r['exports']['FINAL']
  if h['count']!=1260 or not math.isfinite(h['mean_CE']):raise ValueError('Complete finite source readout required')
 main=results['DYNAMIC']['exports']['FINAL'];healthy=all(r['source_health'] for r in results.values())
 active=results['DYNAMIC']['evidence_audit']['aux_positive_calls']>=.9*UPDATES
 contrasts={a:dict(error_difference=main['errors']-results[a]['exports']['FINAL']['errors'],CE_difference=main['mean_CE']-results[a]['exports']['FINAL']['mean_CE']) for a in ARMS[:-1]}
 return dict(qualified=healthy and active and all(v['error_difference']<0 and v['CE_difference']<=0 for v in contrasts.values()),all_source_healthy=healthy,signal_active=active,contrasts=contrasts,target_access=False,goal_completed=False)
def request():return dict(run_id=RUN,model_release=MODEL_RELEASE,model_code_commit=MODEL_COMMIT,dependency_release=DEPENDENCY_RELEASE,fold_release=FOLD_RELEASE,center_release=CENTER_RELEASE,center_code_commit=CENTER_COMMIT,rows=[config(r) for r in rows()],source_only=True)
