"""Only As CE risk weights change; same source-only P43As20 and P70 M2 pool."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_quality_risk70 import design as nominal
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE=nominal.PROJECT,nominal.SOURCE
RUN='20261019-phase1-FcrossRX-boundedRisk-e30-s2026092701-m1-r01'
RELEASE='cvs_FcrossRX_bounded_risk_E30_20261019_r01'
OWNER='codex/root/bounded-risk71'
BASE=Path(PROJECT)/'runs'/RUN
REFERENCE_RUN=nominal.REFERENCE_RUN
NOMINAL_RUN=nominal.RUN
SEED,STEPS,UPDATES=2026092701,222,2220
RX=FIT=(1,3,4,6,8)
ARMS=('M3',)
LOCAL_NN,SMOKE_NN=0,4
FULL_FP32_POLICY,CLASSES,SCENES,VIEWS=nominal.FULL_FP32_POLICY,nominal.CLASSES,nominal.SCENES,nominal.VIEWS
FORMAL_VIEWS,TRUTH,PANEL,SOURCE_CACHE=nominal.FORMAL_VIEWS,nominal.TRUTH,nominal.PANEL,nominal.SOURCE_CACHE
read,write=nominal.read,nominal.write
GAMMA,V=.25,(.15,.40,.25,.20)

def reference_config(arm='B01'):return nominal.reference_config(arm)
def old_config(arm=None):return nominal.old_config()
def nominal_config():return nominal.config(next(r for r in nominal.rows() if r['arm']=='M2'))
def rows():return [dict(arm='M3',row_id='M3-s'+str(SEED),model_seed=SEED)]
def config(row,segment=30):
 if row not in rows() or segment!=30:raise ValueError('Only one sameAs20+10 totalE30 tail')
 c=deepcopy(nominal_config())
 c.update(schema='FcrossRX_bounded_risk71_v1',run_id=RUN,**row,launch_owner=OWNER,
  output_root=str(BASE/row['row_id']/'source'),method='BOUNDED_RISK',
  stage='FcrossRX_fullsource_bounded_risk71',
  training_recipe='Same P70 M2 observations/order/6L/U/As20 reset10; only fixed internal As CE pi replaces nominal p',
  model_initialization='Strict original P43 scratchAs20 student/ownEMA; never P70 final; fresh AdamW and cosine10',
  optimizer_policy='RESET original As20 AdamW; same CENTER_UNITS and new cosine10 as P70 M2',
  baseline_reference='Reuse exact P70 M2 cumulative30 and original P43 NoAs30/As30 all7; no retraining, reprediction or rescoring',
  risk=dict(gamma=GAMMA,v=list(V),nominal_pool='Existing source-only P70 nominal_pool, read-only',
   law='pi=(1-gamma)*p+gamma*v; p_g=existing nominal pool N_g/N before all updates',
   satellite_CE_coefficient=.25,satellite_KL_coefficient=.25,
   KL='Original nominal unweighted sameIQ RDrop; not pi weighted',
   new_channel_calls=0,extra_NN=0,all_labeled_CE=True,new_U_rules=0,
   no_recoverability_truth_claim=True,all_scenes_states_supported=True,
   changed_risk_not_unbiased_nominal=True,own_M2_same_logits_counterfactual=True,
   zero_support='Positive pi on empty observed group is a technical stop, no fallback or resimulation'),
  cost_claim='One new tail2220updates/native3162962NN+source12600+formal1176002+sharedrealCPsmoke4=4351568;0new channel simulation',
  budget_authorization='One new ownP43As20+10 total30; exact matching P70M2 reused; no seed/weight scan; good source-qualified recipe may later separately total50',
  benchmark_context='Exposed benchmark; gamma/v selected only from source history; fixed formal scores never tune/promote/retry',
  goal=dict(fullV_clean_M3_vs_NoAs_min_pp=-.1,pooled_midlow_M3_vs_M2_min_pp=.5,
   source_FAIL_still_all7=True,E50_only_after_source_PASS=True,formal_feedback=False))
 c['training_mechanism'].update(L_forwards_per_sample=6,expected_L_forward_packets=1704960,
  objective='.5 RDordinary128+.5 RDmixed128+.25 RDraw128+.1 ORD128 SupCon+.25(sum pi_g meanCEg - plainAsCE)',
  default_W_joint=True,extra_neural_forward_packets=0)
 return c

def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed same-parent/same-pool/same-order bounded CE risk contract differs')
 return c

def make_args(c,device='cuda:0'):
 validate(c);a=nominal.old.make_args(old_config(),device)
 a.run_id,a.candidate_id,a.output_dir=RUN,c['row_id'],c['output_root']
 a.baseline_ckpt=c['parent_source']+'/final_ssdg.pth'
 if a.teacher_ckpt or a.pseudo_temporal_mode!='batch_neighbor' or not a.pseudo_temporal_gate:raise ValueError('Original ownEMA/nativeU rules required')
 return a
