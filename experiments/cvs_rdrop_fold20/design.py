from pathlib import Path
import math
from experiments.cvs_evidence_responsibility_train20 import design as parent
from experiments.cvs_rx_u_increment20 import design as fold
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write=(getattr(parent,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write'))
RUN='20261015-phase1-SIGTIME-rdrop-sourcefold-s2026092701-m2-r01'
RELEASE='cvs_SIGTIME_rdrop_sourcefold_train20_20261015_r01'
OWNER='codex/root/SIGTIME-rdrop-sourcefold-20261015'
DEPENDENCY_RELEASE=PROJECT+'/releases/'+parent.RELEASE
DEPENDENCY_COMMIT='3a5cff49091deef46cad5d787e5daea7b4c0dfb9'
FOLD_RELEASE=PROJECT+'/releases/cvs_SIGTIME_t2_locked_tailavg_sourcefold_20261015_r01'
SIGNAL_RELEASE=PROJECT+'/releases/cvs_SIGTIME_rdrop_source20_20261015_r01'
SIGNAL_RUN='20261015-phase1-SIGTIME-rdrop-source-e20-clean-manysig-m3-r01'
SEED=2026092701;HELD=1;FIT=(3,4,6,8);EPOCHS=20;UPDATES=4440
ARMS=('TWO_CE','RDROP');BASE=Path(PROJECT)/'runs'/RUN
GATE=dict(primary='RDROP FINAL vs matched TWO_CE FINAL',held_RX=1,errors_strictly_fewer=True,mean_CE_not_higher=True,
    final_training_L_CE_max=.5,readout='Fixed final20 original jointly trained default head only',
    on_pass='Only separately registered independent source-seed confirmation; no automatic formal target release',
    on_fail='Close this fixed recipe; no coefficient/epoch/layer/parent rescue or selective rerun',target_access=False)
def base_config():return parent.config(next(r for r in parent.rows() if r['arm']=='BASE_SSL' and r['model_seed']==SEED))
def config(arm):
 if arm not in ARMS:raise ValueError('Only two fixed matched source arms permitted')
 c=base_config();rid=arm+'-RX1-s'+str(SEED)
 return dict(c,schema='phase1_rdrop_genuine_RX1_training_v1',run_id=RUN,row_id=rid,arm=arm,
    method='cvs_matched_twoCE_rdrop_sourcefold',stage='source_RX_fold_full_training',launch_owner=OWNER,
    output_root=(BASE/rid/'source').as_posix(),test_views=[],readouts=['FINAL'],head_fit_updates=0,
    fold=fold.config()['fold'],source_falsification=GATE,
    training_mechanism=dict(two_same_clean_L_stochastic_forwards=True,paired_CE='mean CE1 CE2, LS0',
      KL='0.5 mean sum (p-q)*(logp-logq), differentiable both',alpha=1. if arm=='RDROP' else 0.,
      active_steps=[1,4440],original_native_dropout=True,extra_parameters=0,extra_optimizer_steps=0,
      extra_L_identity_forwards=4440*128,original_L_forward_packets=4440*128,
      U_recipe='Unchanged oldgate/own EMA Gaussian weak/strong; hidden TX; no new U consistency',
      main_default_head_joint=True,teacher='Own scratch only, original EMA0.999 unchanged; never new loss on teacher',
      primary_class_loss_logged='CE1 remains native train/loss_tx_labeled; pairedCE and KL explicitly logged separately'),
    baseline_reference='Fresh matched TWO_CE; no reuse of one-forward ELEMENT or any full-five-RX weights',
    training_recipe='Original SIG_TIME+CENTER25K/default cosine; scratch20 LS0;13L+7U; RX1 excluded from all training/calibration/cache;222 steps/epoch; mean two clean L CE both arms',
    budget_authorization='User continuing training/loss mechanism goal; all three fixed E20 source derivative parents passed. One fixed seed genuine RX1-held-out matched two-arm E20 falsification.',
    cost_claim='2x20 epochs/8880 optimizer steps; each row extra568320 L identity packet-forwards,396368 U repeated observations;1260 heldL final readout,144 unused source anchors separately disclosed',
    goal=dict(clean_three_seed_mean=.84,source_diagnostic_only=True,goal_completed=False,
      novelty='R-Drop is a published mechanism; current experiment tests training efficacy in CVS, not a global novelty claim'))
def validate(c):
 if c!=config(c.get('arm')):raise ValueError('Fixed two-arm source recipe differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c);a=parent.make_args(base_config(),device)
 a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root']
 a.source_fold_held_rx=HELD;a.source_fold_training_receivers=','.join(map(str,FIT))
 if a.epochs!=20 or a.label_epochs!=13 or a.pseudo_epochs!=7 or a.label_smoothing!=0 or a.batch_size!=128 or a.use_aug or a.use_concat_sat_channel_aug:
  raise ValueError('Clean L and original fixed E20 budget required')
 return a
source_validation=fold.source_validation
def qualify(results):
 if set(results)!=set(ARMS):raise ValueError('Both fixed final rows required')
 for r in results.values():
  m=r['exports']['FINAL']
  if m['count']!=1260 or not math.isfinite(m['mean_CE']):raise ValueError('Complete finite held RX required')
 a,b=(results[k]['exports']['FINAL'] for k in ('RDROP','TWO_CE'))
 healthy=all(r['source_health'] for r in results.values())
 return dict(qualified=healthy and a['errors']<b['errors'] and a['mean_CE']<=b['mean_CE'],all_source_healthy=healthy,
    held_RX=1,error_difference=a['errors']-b['errors'],CE_difference=a['mean_CE']-b['mean_CE'],
    target_access=False,formal_target_qualified=False,goal_completed=False)
def request():return dict(run_id=RUN,dependency_release=DEPENDENCY_RELEASE,fold_release=FOLD_RELEASE,signal_release=SIGNAL_RELEASE,rows=[config(a) for a in ARMS],source_only=True)
