from pathlib import Path
import math
from experiments.cvs_sigtime_rdrop_legacy import design as original
from experiments.cvs_rx_u_increment20 import design as fold
from .mechanism import SITES
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write=(getattr(original,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write'))
RUN='20261015-phase1-SIGTIME-rdrop-paired-masks-sourcefold-s2026092701-m4-r01';RELEASE='cvs_SIGTIME_rdrop_paired_masks_source20_20261015_r01';OWNER='codex/root/SIGTIME-RDrop-paired-masks-20261015'
MODEL_RELEASE=PROJECT+'/releases/cvs_originalSIGTIME_rdrop20_50_20261015_r01';MODEL_COMMIT='b3ac442e2d33a5b315fb3251c30e0ad7083566bc'
DEPENDENCY_RELEASE=original.DEPENDENCY_RELEASE;DEPENDENCY_COMMIT=original.DEPENDENCY_COMMIT;FOLD_RELEASE=original.FOLD_RELEASE
BASE=Path(PROJECT)/'runs'/RUN;SEED=2026092701;FIT=(3,4,6,8);HELD=1;UPDATES=4440
ARMS=('INDEP_CE','ANTI_CE','INDEP_RDROP','ANTI_RDROP')
def rows():return [dict(row_id=a+'-RX1-s'+str(SEED),arm=a,coupling='ANTITHETIC' if a.startswith('ANTI_') else 'INDEPENDENT',model_seed=SEED) for a in ARMS]
def native_config(c):return original.config(next(r for r in original.rows() if r['epochs']==20 and r['mode']==('RDROP' if c['arm'].endswith('RDROP') else 'TWO_CE') and r['model_seed']==SEED))
def config(row):
 if row not in rows():raise ValueError('Fixed2x2 paired-mask source matrix required')
 return dict(native_config(row),schema='SIGTIME_RDrop_paired_masks_v1',run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),method='cvs_rdrop_paired_masks',stage='source_RX_fold_full_training',test_views=[],readouts=['FINAL'],fold=fold.config()['fold'],updates=4440,source_falsification=None,
  training_recipe='Unchanged original SIG_TIME RAW_HZ/normalaug/LS.01/defaultcos joint; scratch20 13L+7oldU,true RX1 exclusion,222steps',
  training_mechanism=dict(arm=row['arm'],coupling=row['coupling'],alpha=1. if row['arm'].endswith('RDROP') else 0.,label_smoothing=.01,
   objective='Exact original same captured augmented L128 two-pass smoothed CE mean plus alpha*symmetricKL',
   mask_rule='First native Bernoulli dropout. INDEP second native Bernoulli. ANTI: if first dropped,second keeps; otherwise conditional second drop p/(1-p),so both marginals p and simultaneous drop0',
   sites=SITES,scope='Only13 actual identity feature dropout sites during paired L. Original U,EMA,domain backbone,eval and all other contexts use native dropout',
   implementation='Native forward scale and exact native CUDA backward; no extra RNG draws,parameters,optimizersteps,oraugmentation redraw',
   interpretation='Negative per-coordinate mask correlation, not a guarantee of lower nonlinear CE gradient variance or matched effective KL/AdamW displacement',default_head_joint=True,KL_default_W_joint=True,extra_parameters=0,extra_optimizer_steps=0,extra_L_identity_forwards=568320),
  source_selection=dict(primary='ANTI_RDROP',controls=list(ARMS[:-1]),held_RX=1,final_L_CE_max=.5,errors_strictly_fewer=True,mean_CE_no_higher=True,mask_marginal_tolerance=.005,antithetic_double_drop=0,error_interaction_strictly_negative=True,on_pass='Separately preregister main only',on_fail='Close fixed mask recipe;no coupling/layer/seed/epoch/alpha rescue',target_access=False),
  baseline_reference='Fresh INDEP_RDROP exactly reproduces native original RDrop; CE controls are paired CE with original hidden-U recipe,not a pureCE-only baseline',
  budget_authorization='Current only SIG_TIME+RDrop; independent mask-sampling hypothesis. Four scratch20 arms,fixed first predefined source seed01 without target parent selection',
  goal=dict(clean_three_seed_mean=.84,source_only=True,target_access=False,global_novelty_claim=False,goal_completed=False))
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed paired masks source config differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c);a=original.make_args(native_config(c),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];a.source_fold_held_rx=HELD;a.source_fold_training_receivers=','.join(map(str,FIT))
 if (a.epochs,a.label_epochs,a.pseudo_epochs,a.label_smoothing,a.use_aug,a.batch_size)!=(20,13,7,.01,True,128) or a.use_concat_sat_channel_aug or a.baseline_ckpt or a.teacher_ckpt:raise ValueError('Original recipe and scratch required')
 return a
source_validation=fold.source_validation
def qualify(results):
 if set(results)!=set(ARMS):raise ValueError('Complete2x2 fixed source matrix required')
 for r in results.values():
  h=r['exports']['FINAL']
  if h['count']!=1260 or not math.isfinite(h['mean_CE']):raise ValueError('Complete finite held source predictions required')
 main=results['ANTI_RDROP']['exports']['FINAL'];healthy=all(z['source_health'] for z in results.values());contrasts={a:dict(error_difference=main['errors']-results[a]['exports']['FINAL']['errors'],CE_difference=main['mean_CE']-results[a]['exports']['FINAL']['mean_CE']) for a in ARMS[:-1]}
 errors={a:r['exports']['FINAL']['errors'] for a,r in results.items()};interaction=(errors['ANTI_RDROP']-errors['INDEP_RDROP'])-(errors['ANTI_CE']-errors['INDEP_CE'])
 return dict(qualified=healthy and interaction<0 and all(v['error_difference']<0 and v['CE_difference']<=0 for v in contrasts.values()),all_source_healthy=healthy,error_interaction=interaction,contrasts=contrasts,target_access=False,goal_completed=False)
def request():return dict(run_id=RUN,model_release=MODEL_RELEASE,model_code_commit=MODEL_COMMIT,dependency_release=DEPENDENCY_RELEASE,fold_release=FOLD_RELEASE,rows=[config(r) for r in rows()],source_only=True)
