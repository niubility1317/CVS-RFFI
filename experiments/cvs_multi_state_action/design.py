"""Immutable report-complete source diagnostics and scratch identity controls."""
from pathlib import Path
from experiments.cvs_multi_disentangle import design as parent
from experiments.cvs_multi_action_audit import design as audit_parent
from experiments.cvs_phase1_stack.source import require_budget
from experiments.cvs_phase1_stack.design import args_from_validated_config
from experiments.cvs_phase1_overlay.contract import read, write

ROOT=Path(__file__).resolve().parents[2]
PROJECT=parent.PROJECT; SOURCE=parent.SOURCE; CAPSULE=parent.CAPSULE; TRUTH=parent.TRUTH
SEEDS=parent.SEEDS; CLASSES=parent.CLASSES; FULL_FP32_POLICY=parent.FULL_FP32_POLICY
RUN='20261009-phase1-multi-state-action-manysig-m36-r01'
RELEASE='cvs_multi_state_action_20261009_r01'
BASE=Path(PROJECT)/'runs'/RUN
OWNER='codex/root/multi-state-action-20261009'
ARMS=('native','real_views','unified','L','LT','LTR','L_EG','LT_EG','LTR_EG')
RECIPE=dict(warmup_epochs=20,auxiliary_every_steps=4,auxiliary_batch_size=32,
 auxiliary_lr=.0002,auxiliary_lr_min=.000001,identity_real_weight=.05,
 identity_virtual_weight=.05,receiver_identity_weight=.01,total_auxiliary_weight=.05,
 paired_consistency_weight=.01,reference_refresh_steps=888,receiver_max_age_steps=888,
 gradient_audit_every_steps=888,epochs=200,steps_per_epoch=222,hidden=48,
 resume=False,source_selection='fixed_E200_all_preregistered_controls',
 reliability='source-heldout minimum of normalized-z and all-competitor-margin skills; clip[0,1]; min16; percondition; reset on reference version',
 model_spec='independent increment/state first+second actions; exact29 reference; packetwiseR distribution',
 deployment='identity_only',interventions=parent.RECIPE['interventions'])
AUDIT=dict(steps=200,batch_size=32,interventions=3,learning_rate=.0002,
 fit_per_tx_rx=64,audit_per_tx_rx=32,physical_split_seed=20261009,
 augmentation_seed=2026100901,evaluation_seed=2026100911,
 exposures=['clean','source_practical_mid'],modes=['p_h_first_all','p_hs_first_all',
 'p_hs_second_all','p_hs_second_exact','analytic_hs_second_exact','neural_hs_second_exact'],
 heldout_tx=list(range(6)),heldout_modes=['p_hs_second_exact','neural_hs_second_exact'],
 source_admission='Report every branch/condition vs zero and directedRXmean; efficacy never implied by positive hskill alone. Online independent calibration gates actual virtual loss; no target access.',
 no_early_stopping=True)

def rows():
 return [dict(stage='multi_state_action',arm=a,model_seed=s,row_id=a+'-s'+str(s)) for s in SEEDS for a in ARMS]

def audit_rows():
 return [dict(row_id='audit-s'+str(s),model_seed=s,kind='audit') for s in SEEDS]

def config(row):
 c=parent.config(dict(stage='multi_disentangle',arm='legacy_cosine',model_seed=row['model_seed'],row_id='legacy_cosine-s'+str(row['model_seed'])))
 c.pop('multi_recipe',None)
 c.update(schema='multi_state_action_v1',run_id=RUN,**row,method='cvs_multi_state_action',
  output_root=(BASE/row['row_id']/'source').as_posix(),state_recipe=RECIPE,
  virtual_gradient_path='EG' if row['arm'].endswith('_EG') else 'G_only',checkpoint_sources=[])
 return c

def audit_config(row):
 p=audit_parent.parent_config(row['model_seed'])
 return dict(schema='multi_state_action_audit_v1',run_id=RUN,**row,recipe=AUDIT,
  checkpoint=(Path(p['output_root'])/'final_ssdg.pth').as_posix(),parent_config=p,
  source_contract=SOURCE,output_root=(BASE/row['row_id']).as_posix(),target_access=False)

def validate(c):
 r=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if r is None or config(r)!=c:raise ValueError('Unregistered identity configuration')
 return c

def validate_audit(c):
 r=next((r for r in audit_rows() if r['row_id']==c.get('row_id')),None)
 if r is None or audit_config(r)!=c:raise ValueError('Unregistered audit configuration')
 return c

def make_args(c,device='cuda:0'):
 validate(c);a=args_from_validated_config(c,device)
 a.pseudo_domain_gate=False;a.pseudo_temporal_mode='batch_neighbor';a.pseudo_temporal_bank_min_streak=2
 a.phase1_lr_schedule='cosine';a.phase1_lr_min=1e-6
 return a
