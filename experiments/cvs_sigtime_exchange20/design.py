from pathlib import Path
import math
from experiments.cvs_sigtime_rdrop_legacy import design as original
from experiments.cvs_rx_u_increment20 import design as fold
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write=(getattr(original,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write'))
RUN='20261015-phase1-originalSIGTIME-residual-exchange-sourcefold-s2026092702-m3-r01'
RELEASE='cvs_originalSIGTIME_residual_exchange_source20_20261015_r01'
OWNER='codex/root/original-SIGTIME-residual-exchange-20261015'
MODEL_RELEASE=PROJECT+'/releases/cvs_originalSIGTIME_rdrop20_50_20261015_r01'
MODEL_COMMIT='b3ac442e2d33a5b315fb3251c30e0ad7083566bc'
DEPENDENCY_RELEASE=original.DEPENDENCY_RELEASE
DEPENDENCY_COMMIT=original.DEPENDENCY_COMMIT
FOLD_RELEASE=original.FOLD_RELEASE
BASE=Path(PROJECT)/'runs'/RUN
SEED=2026092702;FIT=(3,4,6,8);HELD=1;UPDATES=4440;ARMS=('SELF','SAME_RX','CROSS_RX')

def native_config():
 return original.config(next(r for r in original.rows() if r['epochs']==20 and r['mode']=='RDROP' and r['model_seed']==SEED))

def rows():return [dict(row_id=a+'-RX1-s'+str(SEED),arm=a,model_seed=SEED) for a in ARMS]

def config(row):
 if row not in rows():raise ValueError('Only fixed seed02 source conditions permitted')
 return dict(native_config(),schema='original_SIGTIME_conditional_residual_exchange_v1',run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),
  method='cvs_conditional_dropout_residual_exchange',stage='source_RX_fold_full_training',test_views=[],readouts=['FINAL'],fold=fold.config()['fold'],
  training_recipe='Original SIG_TIME RAW_HZ/normalaug/LS.01/defaultcos joint; own scratch20 13L+7oldU; RX1 excluded before model/ownEMA/calibration/cache;222steps',
  training_mechanism=dict(center='(z1+z2)/2',residual='(z1-z2)/2',virtual='center_i +/- residual_donor_i',
   grouping='Visible L TX/day/eq, first2 per eligible RX cell, >=2 eligible RX; others SELF; same cohort for all conditions',
   donor=row['arm'],SAME_RX='swap the two within-RX samples',CROSS_RX='cycle sorted eligible RX cells, preserve slot',
   objective='SELF exact original RDrop correction; others smoothed CE of virtual +/- minus native CE1 plus original symmetricKL',alpha=1.,label_smoothing=.01,
   input='Two native stochastic passes reuse identical captured post-augmentation L128 IQ; no augmentation redraw',
   gradient='Both centers and donor differences, native feature gains and original W differentiable; no detach',
   preservation='Per-packet center and pooled class/day/eq two-view energy; no covariance fitting or inferred pure RX residual',
   default_head_joint=True,extra_parameters=0,extra_optimizer_steps=0,extra_L_identity_forwards=568320,
   U_recipe='Original hidden U/ownEMA/batch_neighbor unchanged; no new U consistency',sinc_coordinates='RAW_HZ'),
  source_selection=dict(primary='CROSS_RX',controls=['SELF','SAME_RX'],held_RX=1,readout='Fixed final20 native default head on all1260 held source L',
   final_L_CE_max=.5,minimum_swapped_fraction=.05,errors_strictly_fewer=True,mean_CE_no_higher=True,
   on_pass='Separate preregistered next main matrix only; no automatic target release',on_fail='Close fixed operator; no coefficient/layer/seed/pairing/epoch rescue',target_access=False),
  budget_authorization='Active Phase1 mechanism goal; independent source-only falsification of pre-existing deferred idea. Seed02 fixed in prototype before current target summary; three scratch E20 conditions, no inherited weights.',
  baseline_reference='Fresh SELF exact original RDrop on same raw/augmented source fold; historical scores never reused as matched controls',
  cost_claim='3scratch20/60model-epoch/13320updates;each568320extraL,396368hiddenU observations,1260heldL,144unused source anchors;0target',
  goal=dict(clean_three_seed_mean=.84,source_only=True,target_access=False,global_novelty_claim=False,goal_completed=False))

def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed original recipe/source exchange matrix differs')
 return c

def make_args(c,device='cuda:0'):
 validate(c);a=original.make_args(native_config(),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root']
 a.source_fold_held_rx=HELD;a.source_fold_training_receivers=','.join(map(str,FIT))
 if (a.epochs,a.label_epochs,a.pseudo_epochs,a.label_smoothing,a.use_aug,a.batch_size)!=(20,13,7,.01,True,128) or a.use_concat_sat_channel_aug or a.baseline_ckpt or a.teacher_ckpt:raise ValueError('Original recipe and own scratch initialization required')
 return a
source_validation=fold.source_validation

def qualify(results):
 if set(results)!=set(ARMS):raise ValueError('All three final source rows required')
 for r in results.values():
  v=r['exports']['FINAL']
  if v['count']!=1260 or not math.isfinite(v['mean_CE']):raise ValueError('Complete finite held RX1 readout required')
 main=results['CROSS_RX']['exports']['FINAL'];healthy=all(r['source_health'] for r in results.values());signal=results['CROSS_RX']['mean_swapped_fraction']>=.05
 contrasts={a:dict(error_difference=main['errors']-results[a]['exports']['FINAL']['errors'],CE_difference=main['mean_CE']-results[a]['exports']['FINAL']['mean_CE']) for a in ('SELF','SAME_RX')}
 return dict(qualified=healthy and signal and all(v['error_difference']<0 and v['CE_difference']<=0 for v in contrasts.values()),all_source_healthy=healthy,signal_active=signal,contrasts=contrasts,target_access=False,goal_completed=False)

def request():return dict(run_id=RUN,model_release=MODEL_RELEASE,model_code_commit=MODEL_COMMIT,dependency_release=DEPENDENCY_RELEASE,fold_release=FOLD_RELEASE,rows=[config(r) for r in rows()],source_only=True)
