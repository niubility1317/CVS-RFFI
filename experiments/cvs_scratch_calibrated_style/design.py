"""Source-motivated factorial: calibration time x joint gradient control."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_scratch_boundary_style import design as old
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,SEEDS,FULL_FP32_POLICY=(getattr(old,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','SEEDS','FULL_FP32_POLICY'))
read,write,baseline=old.read,old.write,old.baseline
RUN='20261012-phase1-scratch-calibrated-style-manysig-m36-r01'
RELEASE='cvs_scratch_calibrated_style_20261012_r01';OWNER='codex/root/scratch-calibrated-style-20261012';BASE=Path(PROJECT)/'runs'/RUN
BASE_ARMS=('M_SHALLOW','M_MIDDLE','M_BOTH','JOINT')
POLICIES=('DELAY','CAP','DELAYCAP')
ARMS=tuple(f'{policy}_{base}' for policy in POLICIES for base in BASE_ARMS)
PRIMARY='DELAYCAP_JOINT'
PAIRS=[(f'DELAYCAP_{base}',f'{policy}_{base}') for base in BASE_ARMS for policy in ('DELAY','CAP')]
PAIRS += [(f'{policy}_M_BOTH',f'{policy}_{base}') for policy in POLICIES for base in ('M_SHALLOW','M_MIDDLE')]
PAIRS += [(f'{policy}_JOINT',f'{policy}_M_BOTH') for policy in POLICIES]
PAIRS += [(arm,'P0') for arm in ARMS]
REFERENCE_RUN=old.RUN
REFERENCE_PAIRS=[(arm,arm.split('_',1)[1]) for arm in ARMS]+[(arm,base) for arm in ARMS for base in ('CE','SSL')]

def rows():return [dict(arm=a,model_seed=s,row_id=f'{a}-s{s}') for s in SEEDS for a in ARMS]

def config(row):
 if row not in rows():raise ValueError('Unregistered calibrated-style row')
 policy,base=row['arm'].split('_',1)
 prior=next(r for r in old.rows() if r['arm']==base and r['model_seed']==row['model_seed'])
 c=deepcopy(old.config(prior))
 c.update(schema='scratch_calibrated_style_v1',run_id=RUN,**row,base_arm=base,calibration_policy=policy,
  method='cvs_scratch_calibrated_style',stage='phase1_scratch_calibrated_style',
  output_root=(BASE/row['row_id']/'source').as_posix(),launch_owner=OWNER)
 c['teacher']['cache_root']=(BASE/'peer_cache').as_posix()
 c['evaluation']['primary_arm']=PRIMARY
 c['multilevel'].update(calibration_after_updates=444 if policy in ('DELAY','DELAYCAP') else 0,
  joint_gradient_cap=dict(enabled=policy in ('CAP','DELAYCAP'),rho=.2,eps=1e-12),
  gradient_budget_scope='fixed 12-pair own-margin calibration at registered update; optional detached single-scalar joint cap on complete identity parameter partition')
 return c

def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Calibrated-style configuration differs')
 return c

def learning_rate(c,epoch):return old.learning_rate(c,epoch)
