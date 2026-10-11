from pathlib import Path
from experiments.cvs_stage2_rx_risk import design as parent
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SEEDS=parent.PROJECT,parent.SEEDS
read,write=parent.read,parent.write
RUN='20261011-phase1-u-geometry-audit-manysig-m3-r01'
RELEASE='cvs_u_geometry_audit_20261011_r01'
BASE=Path(PROJECT)/'runs'/RUN
OWNER='codex/root/u-geometry-audit-20261011'

def rows():return [dict(row_id=f'E200-s{s}',model_seed=s) for s in SEEDS]

def config(row):
 if row not in rows():raise ValueError('Unregistered geometry row')
 pc=parent.config(next(r for r in parent.rows() if r['arm']=='PHYS' and r['model_seed']==row['model_seed']))
 return dict(run_id=RUN,**row,output=(BASE/row['row_id']).as_posix(),parent_build_config=pc,
  source_checkpoint=pc['parent_checkpoint'],launch_owner=OWNER,source_only=True,target_access=False,identity_updates=0,
  roles=dict(L=6300,U=56700,V_used=0),views=['noise','physical_noise'],noise_std=.015,
  confidence_bins=[0.,.95,.99,.999,1.000001],batch_size=256,
  gradient_batches=30,gradient_batch_size=128,
  sampling='All L/U physical IDs once per view; independent RX/day-balanced L/U streams for gradient probes; no U TX access',
  losses=dict(KL='mean accepted KL(stopgrad softmax(clean), altered); fixed batch denominator, confidence >=.95',
   margin='0.5 mean squared centered-logit change, no confidence mask',
   feature='mean squared L2 of unit feature change, no confidence mask'),
  scope='Original E200 eval-mode source sensitivity and gradients, no optimizer or persistent update; not unseen-RX validation or evidence of training benefit',
  selection='All fixed views/losses/seeds reported; no V, target, calibration fitting or automatic candidate selection')

def validate(c):
 r=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if r is None or c!=config(r):raise ValueError('Geometry audit config differs')
 return c
