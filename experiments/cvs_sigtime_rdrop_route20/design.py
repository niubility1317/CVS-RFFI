from pathlib import Path
import math
from experiments.cvs_sigtime_rdrop_legacy import design as original
from experiments.cvs_rx_u_increment20 import design as fold
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write=(getattr(original,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write'))
RUN='20261015-phase1-SIGTIME-rdrop-difference-gradient-sourcefold-s2026092702-m4-r01'
RELEASE='cvs_SIGTIME_rdrop_difference_gradient_source20_20261015_r01'
OWNER='codex/root/SIGTIME-RDrop-difference-gradient-20261015'
MODEL_RELEASE=PROJECT+'/releases/cvs_originalSIGTIME_rdrop20_50_20261015_r01'
MODEL_COMMIT='b3ac442e2d33a5b315fb3251c30e0ad7083566bc'
DEPENDENCY_RELEASE=original.DEPENDENCY_RELEASE;DEPENDENCY_COMMIT=original.DEPENDENCY_COMMIT;FOLD_RELEASE=original.FOLD_RELEASE
BASE=Path(PROJECT)/'runs'/RUN;SEED=2026092702;FIT=(3,4,6,8);HELD=1;UPDATES=4440
ARMS=('FULL_KL','FEATURE_KL','SHRINK','DIFFERENCE')
def native_config():return original.config(next(r for r in original.rows() if r['epochs']==20 and r['mode']=='RDROP' and r['model_seed']==SEED))
def rows():return [dict(row_id=a+'-RX1-s'+str(SEED),arm=a,model_seed=SEED) for a in ARMS]
def config(row):
 if row not in rows():raise ValueError('Only four fixed RDrop source conditions permitted')
 return dict(native_config(),schema='SIGTIME_RDrop_difference_gradient_v1',run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),
  method='cvs_rdrop_difference_gradient',stage='source_RX_fold_full_training',test_views=[],readouts=['FINAL'],fold=fold.config()['fold'],updates=4440,source_falsification=None,
  training_recipe='Unchanged original SIG_TIME RAW_HZ/normalaug/LS.01/defaultcos joint; scratch20 13L+7oldU, true RX1 exclusion,222steps',
  training_mechanism=dict(arm=row['arm'],forward='Identical original two stochastic passes on same captured augmented L128; mean smoothed CE plus symmetricKL alpha1',
   FULL_KL='Original KL gradients to features and W',FEATURE_KL='KL W detached; CE still updates both features and native W',
   SHRINK='Feature KL backward scaled to its own antisymmetric projection norm',
   DIFFERENCE='Feature KL backward (g1-g2)/2,(g2-g1)/2; zero common feature-gradient component',
   scope='Only KL graph changes. CE/default classifier/SIG_TIME/U/EMA/dropout rates/augmentation/schedule stay unchanged',
   interpretation='Zero common component at joint160 gradient, not a guarantee of unchanged encoder mean after AdamW',
   norm_control='SHRINK matches DIFFERENCE norm locally at its own incoming gradients; not matched parameter displacement across trajectories',
   alpha=1.,label_smoothing=.01,cosine_scale=30.,cosine_eps=.0001,extra_parameters=0,extra_optimizer_steps=0,extra_L_identity_forwards=568320,
   default_head_joint=True,original_hidden_U_recipe_unchanged=True,augmentation_redraws=0),
  source_selection=dict(primary='DIFFERENCE',controls=['FULL_KL','FEATURE_KL','SHRINK'],held_RX=1,readout='Fixed final20 default head on1260 held source L',
   final_L_CE_max=.5,minimum_removed_common_energy_fraction=.01,errors_strictly_fewer=True,mean_CE_no_higher=True,
   on_pass='Separately preregister main only; no automatic target access',on_fail='Close fixed gradient operator; no coefficient/epoch/seed/layer rescue',target_access=False),
  baseline_reference='Fresh FULL_KL exact original RDrop; no historical model or result reuse',
  budget_authorization='User now explicitly optimizes SIG_TIME plus RDrop only, emphasizing RDrop mechanism. One new gradient hypothesis with three controls, single fixed source seed02, four scratch E20 rows.',
  cost_claim='4scratch20/80modelepoch/17760updates;each568320extraL,396368hiddenU observations,1260heldL,144unused source anchors;0target',
  goal=dict(clean_three_seed_mean=.84,source_only=True,target_access=False,global_novelty_claim=False,goal_completed=False))
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed RDrop routing/source matrix differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c);a=original.make_args(native_config(),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];a.source_fold_held_rx=HELD;a.source_fold_training_receivers=','.join(map(str,FIT))
 if (a.epochs,a.label_epochs,a.pseudo_epochs,a.label_smoothing,a.use_aug,a.batch_size)!=(20,13,7,.01,True,128) or a.use_concat_sat_channel_aug or a.baseline_ckpt or a.teacher_ckpt:raise ValueError('Unchanged original RDrop recipe/own scratch required')
 return a
source_validation=fold.source_validation
def qualify(results):
 if set(results)!=set(ARMS):raise ValueError('All four final source rows required')
 for r in results.values():
  v=r['exports']['FINAL']
  if v['count']!=1260 or not math.isfinite(v['mean_CE']):raise ValueError('Complete finite heldRX readout required')
 main=results['DIFFERENCE']['exports']['FINAL'];healthy=all(r['source_health'] for r in results.values());signal=results['DIFFERENCE']['removed_common_energy_fraction']>=.01
 contrasts={a:dict(error_difference=main['errors']-results[a]['exports']['FINAL']['errors'],CE_difference=main['mean_CE']-results[a]['exports']['FINAL']['mean_CE']) for a in ARMS[:-1]}
 return dict(qualified=healthy and signal and all(v['error_difference']<0 and v['CE_difference']<=0 for v in contrasts.values()),all_source_healthy=healthy,signal_active=signal,contrasts=contrasts,target_access=False,goal_completed=False)
def request():return dict(run_id=RUN,model_release=MODEL_RELEASE,model_code_commit=MODEL_COMMIT,dependency_release=DEPENDENCY_RELEASE,fold_release=FOLD_RELEASE,rows=[config(r) for r in rows()],source_only=True)
