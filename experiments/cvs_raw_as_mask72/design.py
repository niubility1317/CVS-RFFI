"""Only raw--As base160 dropout correlation changes; fixed full practical and sixL."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_quality_risk70 import design as nominal
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE=nominal.PROJECT,nominal.SOURCE
RUN='20261019-phase1-FcrossRX-rawAsMask-e30-s2026092701-m2-r01'
RELEASE='cvs_FcrossRX_rawAsMask_E30_20261019_r01'
OWNER='codex/root/raw-as-mask72'
BASE=Path(PROJECT)/'runs'/RUN
REFERENCE_RUN,NOMINAL_RUN=nominal.REFERENCE_RUN,nominal.RUN
SEED,STEPS,UPDATES=2026092701,222,2220
RX=FIT=(1,3,4,6,8)
ARMS=('I1','S1')
LOCAL_NN,SMOKE_NN=0,4
FULL_FP32_POLICY,CLASSES,SCENES,VIEWS=nominal.FULL_FP32_POLICY,nominal.CLASSES,nominal.SCENES,nominal.VIEWS
FORMAL_VIEWS,TRUTH,PANEL,SOURCE_CACHE=nominal.FORMAL_VIEWS,nominal.TRUTH,nominal.PANEL,nominal.SOURCE_CACHE
read,write=nominal.read,nominal.write
MASK_SEED=20261019072

def reference_config(arm='B01'):return nominal.reference_config(arm)
def old_config(arm=None):return nominal.old_config()
def nominal_config():return nominal.config(next(r for r in nominal.rows() if r['arm']=='M1'))
def rows():return [dict(arm=a,row_id=a+'-s'+str(SEED),model_seed=SEED) for a in ARMS]
def config(row,segment=30):
 if row not in rows() or segment!=30:raise ValueError('Only two ownAs20+10 totalE30 tails')
 c=deepcopy(old_config());pool=deepcopy(nominal_config()['quality_pool'])
 c.update(schema='FcrossRX_raw_as_mask72_v1',run_id=RUN,**row,launch_owner=OWNER,
  output_root=str(BASE/row['row_id']/'source'),stage='FcrossRX_fullsource_raw_as_mask72',
  method='MASK_INDEPENDENT' if row['arm']=='I1' else 'MASK_SHARED',
  training_recipe='Same original sixL/As CE and KL/nativeU; only raw-As base160 fusion dropout correlation differs',
  model_initialization='Strict original P43 scratchAs20 student/ownEMA; never P70 or P71 final',
  optimizer_policy='RESET original As20 AdamW/lr2e-4/wd1e-4/cos10/CENTER_UNITS',
  quality_pool=pool,
  ordered_pool=dict(root=pool['root'],observations=142080,order='Original generation order64*(step-1):64*step',
   current_L_As_physical_pair_required=True,one_use=True,new_simulations=0,no_quality_weighting=True),
  mask_coupling=dict(mode='INDEPENDENT_FUSE' if row['arm']=='I1' else 'SHARED_RAW_AS_FUSE',
   module='id_backbone.encoder.core.id_backbone.fuse.2',p=.45,width=160,seed=MASK_SEED,
   scope='Only base160 before PA/reference/SIG_TIME addition, not the complete joint embedding',
   views='As64 in mixed128 and corresponding raw64 in raw128',RDrop_pairs_independent=True,
   native_global_RNG_draws_preserved=True,ordinary_and_U_unchanged=True,
   mask_streams='Same private As/raw draws in both arms; S1 only substitutes As mask for paired raw mask',
   extra_NN=0,extra_parameters=0,persistent_mask_state=0,changed_expected_per_view_risk=False),
  baseline_reference='Reuse original P43 NoAs30/As30; two new matched mask arms independent from sameAs20',
  cost_claim='Two new10tails/4440updates,eachnative3162962+source12600+formal1176002,sharedsmoke4=8703132newNN',
  budget_authorization='Two matched ownAs20+10 total30,not E200;sourcePASS required before later total50',
  benchmark_context='Exposed benchmark; formal scores never tune/promote/retry or set mask seed',
  goal=dict(fullV_clean_shared_vs_NoAs_min_pp=-.1,pooled_midlow_shared_vs_independent_min_pp=.5,
   source_FAIL_still_all7=True,E50_only_after_source_PASS=True,formal_feedback=False))
 c['training_mechanism'].update(L_forwards_per_sample=6,expected_L_forward_packets=1704960,
  objective='.5 RDordinary128+.5 RDmixed128+.25 RDraw128+.1 ORD128 SupCon; AsCE=.25,AsKL=.25 unchanged',
  default_W_joint=True,extra_neural_forward_packets=0)
 return c

def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed raw-As single-layer mask-correlation contract differs')
 return c

def make_args(c,device='cuda:0'):
 validate(c);a=nominal.old.make_args(old_config(),device)
 a.run_id,a.candidate_id,a.output_dir=RUN,c['row_id'],c['output_root']
 a.baseline_ckpt=c['parent_source']+'/final_ssdg.pth'
 if a.teacher_ckpt or a.pseudo_temporal_mode!='batch_neighbor' or not a.pseudo_temporal_gate:raise ValueError('Original ownEMA/nativeU rules required')
 return a
