from pathlib import Path
import math
from experiments.cvs_evidence_responsibility_train20 import design as parent
from experiments.cvs_rx_u_increment20 import design as fold
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write=(getattr(parent,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write'))
RUN='20261015-phase1-SIGTIME-dropout-residual-sourcefold-s2026092702-m3-r01'
RELEASE='cvs_SIGTIME_dropout_residual_source20_20261015_r01'
OWNER='codex/root/SIGTIME-dropout-residual-source20-20261015'
DEPENDENCY_RELEASE=PROJECT+'/releases/'+parent.RELEASE
DEPENDENCY_COMMIT='3a5cff49091deef46cad5d787e5daea7b4c0dfb9'
FOLD_RELEASE=PROJECT+'/releases/cvs_SIGTIME_t2_locked_tailavg_sourcefold_20261015_r01'
BASE=Path(PROJECT)/'runs'/RUN;SEED=2026092702;FIT=(3,4,6,8);HELD=1;UPDATES=4440
ARMS=('SELF','SAME_RX','CROSS_RX')
def base_config():return parent.config(next(r for r in parent.rows() if r['arm']=='BASE_SSL' and r['model_seed']==SEED))
def rows():return [dict(row_id=arm+'-RX1-s'+str(SEED),arm=arm,model_seed=SEED) for arm in ARMS]
def config(row):
 if row not in rows():raise ValueError('Only three fixed seed02 source-fold conditions permitted')
 return dict(base_config(),schema='phase1_dropout_residual_exchange_source20_v1',run_id=RUN,**row,output_root=(BASE/row['row_id']/'source').as_posix(),
  method='cvs_stochastic_residual_exchange',launch_owner=OWNER,stage='source_RX_fold_full_training',test_views=[],readouts=['FINAL'],head_fit_updates=0,fold=fold.config()['fold'],
  training_recipe='SIG_TIME+CENTER25K/native default cosine joint; scratch20 LS0/13L+7oldU; RX1 excluded before student/ownEMA/calibration/cache;222steps;two same clean L stochastic forwards',
  training_mechanism=dict(center='(z1+z2)/2',residual='(z1-z2)/2',virtual='center_i plus/minus residual_donor_i',
   grouping='Visible L TX/eq/day; RX cells with >=2 entries, first2 each; at least2 RX; all other entries SELF',
   donor=row['arm'],SAME_RX='swap two entries within each eligible RX cell',CROSS_RX='cycle sorted eligible RX cells, preserve slot',
   objective='mean CE(defaultW,virtual_plus/minus,y)+original symmetricKL(native logits1,logits2)',alpha=1.,
   gradient='Both stochastic features, centers, donor residuals, all existing feature gains and native W live; no detach',
   scale='Native cosine30; no fitted head, new coefficient, residual normalization, noise clipping, or extra parameters',
   moment_invariants='Per-packet two-view center and class/eq/day pooled residual energy preserved; not physical hardware counterfactual',
   extra_L_identity_forwards=568320,extra_parameters=0,extra_optimizer_steps=0,original_native_dropout=True,
   U_recipe='Unchanged original source-hidden U/ownEMA/batch_neighbor/strong/no new U consistency',main_default_head_joint=True),
  source_selection=dict(primary='CROSS_RX',controls=['SELF','SAME_RX'],held_RX=1,fixed_final=True,
   qualify='All finalLCE<=.5; CROSS_RX mean swapped fraction>=.05; CROSS_RX errors strictly fewer and CE no higher than BOTH controls on1260 held source L',
   on_pass='Separately preregister next fixed main matrix; no automatic target/student release',on_fail='Close fixed operator; no coefficient/layer/seed/epoch/pairing rescue',target_access=False),
  budget_authorization='User active Phase1 training-mechanism goal and full E20/E50 comparisons; one new hypothesis, first3 matched scratch source E20 conditions only',
  baseline_reference='Fresh SELF, exact RDrop in real arithmetic; no historical weights/results reused as control',
  cost_claim='3scratch20/60modelepoch/13320updates;2L forward passes/update, each568320extraL; each396368hiddenU observations;held1260+144unused source anchors;0target',
  goal=dict(clean_three_seed_mean=.84,source_only=True,target_access=False,global_novelty_claim=False,goal_completed=False))
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed source exchange/roles/budget/defaulthead differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c);a=parent.make_args(base_config(),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];a.source_fold_held_rx=1;a.source_fold_training_receivers=','.join(map(str,FIT))
 if a.epochs!=20 or a.label_epochs!=13 or a.pseudo_epochs!=7 or a.label_smoothing!=0 or a.batch_size!=128 or a.use_aug or a.use_concat_sat_channel_aug:raise ValueError('Original source E20 SSL recipe required')
 return a
source_validation=fold.source_validation
def qualify(results):
 if set(results)!=set(ARMS):raise ValueError('All three final source rows required')
 for r in results.values():
  if r['exports']['FINAL']['count']!=1260 or not math.isfinite(r['exports']['FINAL']['mean_CE']):raise ValueError('Full finite held RX1 source readout required')
 main=results['CROSS_RX']['exports']['FINAL'];healthy=all(r['source_health'] for r in results.values());signal=results['CROSS_RX']['mean_swapped_fraction']>=.05
 contrasts={a:dict(error_difference=main['errors']-results[a]['exports']['FINAL']['errors'],CE_difference=main['mean_CE']-results[a]['exports']['FINAL']['mean_CE']) for a in ('SELF','SAME_RX')}
 return dict(qualified=healthy and signal and all(v['error_difference']<0 and v['CE_difference']<=0 for v in contrasts.values()),all_source_healthy=healthy,signal_active=signal,contrasts=contrasts,target_access=False,goal_completed=False)
def request():return dict(run_id=RUN,dependency_release=DEPENDENCY_RELEASE,fold_release=FOLD_RELEASE,rows=[config(r) for r in rows()],source_only=True)
