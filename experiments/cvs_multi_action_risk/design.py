"""Fixed three-seed controls; the native baseline is genuinely clean CE only."""
from pathlib import Path
from copy import deepcopy
from experiments.cvs_legacy_no_sat import design as parent
from experiments.cvs_phase1_stack.design import args_from_validated_config
from experiments.cvs_phase1_stack.source import require_budget
from experiments.cvs_phase1_overlay.contract import read,write

ROOT=Path(__file__).resolve().parents[2]
PROJECT=parent.PROJECT;SOURCE=parent.SOURCE;CAPSULE=parent.CAPSULE;TRUTH=parent.TRUTH
CLASSES=parent.CLASSES;FULL_FP32_POLICY=parent.FULL_FP32_POLICY
SEEDS=(2026092701,2026092702,2026092703)
RUN='20261010-phase1-multi-action-risk-manysig-m48-r01'
RELEASE='cvs_multi_action_risk_20261010_r01'
BASE=Path(PROJECT)/'runs'/RUN
OWNER='codex/root/multi-action-risk-20261010'
ARMS=('native','random_ce','random_cons','exact_g','L','LT','LR','LTR','LTR_mean','shared_LTR',
      'L_budget','LT_budget','LR_budget','LT_no_edges','LT_label_free','LT_label_free_U')
RECIPE=dict(warmup_epochs=20,auxiliary_every_steps=4,auxiliary_batch_size=32,
 auxiliary_lr=.0002,auxiliary_lr_min=.000001,reference_refresh_steps=888,
 gradient_audit_every_steps=888,epochs=200,steps_per_epoch=222,hidden=48,
 branch_weights=dict(linear=.015,temporal=.015,joint=.02,receiver=.01),
 proposal_candidates=4,random_fraction=.25,receiver_tail_alpha=.25,
 proposal_risk_relative_tolerance=.25,proposal_risk_error_ema=.9,proposal_quality_bins=4,
 proposal_min_distinct_audit_ids=16,receiver_live_audit_reencode_max=32,
 auxiliary_batch_scope='at most32 fresh source-L packets; R may separately re-encode at most32 cached audit IQ packets; count both',
 receiver_max_age_steps=888,resume=False,source_selection='fixed_E200_all_preregistered_controls',
 deployment='identity_only',reference='periodically frozen own student; no EMA teacher',
 action_scope='bounded post-receive linear/phase increments; no satellite simulator in training',
 source_u_scope='only isolated LT_label_free_U auxiliary endpoint fitting; no metadata/identity pseudo loss')
STRESS=dict(seed=2026101007,physical_selection='all existing source V',batch_size=256,
 views=['clean','linear','temporal','LT','TL','noise_fixed_LT','LTL_heldout','curvature_pressure'],
 clean_noninferiority_pp=.5,noise_tie_pp=.2,tail_alpha=.25,
 ranking='clean feasibility then stress mean+worst RX risk; descriptive source-only ranking across three seeds; within .2pp prefer fewer auxiliary parameters')
DIAGNOSTIC=dict(seed=2026101011,fit_per_tx_rx=32,audit_per_tx_rx=16,
 steps=120,batch_size=32,heldout_tx=list(range(6)),noise_snr_db=[40,25,15],
 scope='source-L independent auxiliary packet and TX holdouts; native source-trained identity frozen')

def plan(arm):
 p=dict(paths=[],action_mode='proposal',shared=False,r_target='distribution',consistency_weight=0.,
        action_label_free=False,source_u_fit=False,replacements=[],conditional_edges=True)
 if arm=='native':return p
 if arm in ('random_ce','random_cons','exact_g'):
  p.update(action_mode='exact_g' if arm=='exact_g' else 'random',
           replacements=['linear','temporal','joint'],consistency_weight=.01 if arm=='random_cons' else 0.)
  return p
 base=arm.removesuffix('_budget')
 if base=='L':p['paths']=['linear']
 elif base=='LR':p['paths']=['linear','receiver']
 elif base.startswith('LT') or base=='shared_LTR':
  p['paths']=['linear','temporal']+(['receiver'] if base in ('LTR','LTR_mean','shared_LTR') else [])
 else:raise ValueError(arm)
 if arm.endswith('_budget'):
  p['replacements']=[k for k in ('linear','temporal','receiver') if k not in p['paths']]
  if not all(k in p['paths'] for k in ('linear','temporal')):p['replacements'].append('joint')
 p.update(shared=arm=='shared_LTR',r_target='mean' if arm=='LTR_mean' else 'distribution',
          conditional_edges=arm!='LT_no_edges',action_label_free=arm in ('LT_label_free','LT_label_free_U'),
          source_u_fit=arm=='LT_label_free_U')
 return p

def rows():return [dict(stage='multi_action_risk',arm=a,model_seed=s,row_id=a+'-s'+str(s)) for s in SEEDS for a in ARMS]
def config(row):
 c=parent.config(dict(stage='repair',arm='legacy_cosine_no_sat',model_seed=row['model_seed'],row_id='legacy_cosine_no_sat-s'+str(row['model_seed'])))
 c.update(schema='multi_action_risk_v1',run_id=RUN,**row,method='cvs_multi_action_risk',features=[],
          output_root=(BASE/row['row_id']/'source').as_posix(),risk_recipe=deepcopy(RECIPE),arm_plan=plan(row['arm']),
          source_stress=deepcopy(STRESS),checkpoint_sources=[],model_initialization='scratch')
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or config(row)!=c:raise ValueError('Unregistered or changed risk configuration')
 return c
def make_args(c,device='cuda:0'):
 validate(c);a=args_from_validated_config(c,device)
 a.label_smoothing=0.;a.use_unlabeled=False;a.use_ema_teacher=False
 a.lambda_u=0.;a.lambda_ent=0.;a.label_epochs=200;a.pseudo_epochs=0
 a.pseudo_domain_gate=False;a.pseudo_temporal_gate=False
 a.pseudo_strong_agreement=False
 a.phase1_lr_schedule='cosine';a.phase1_lr_min=1e-6
 return a
