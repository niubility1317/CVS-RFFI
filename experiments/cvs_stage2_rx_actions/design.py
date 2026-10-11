from pathlib import Path
from copy import deepcopy
from experiments.cvs_stage2_teacher_factorial import design as previous
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES=(getattr(previous,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES'))
SEEDS=previous.SEEDS;FULL_FP32_POLICY=previous.FULL_FP32_POLICY
read,write=previous.read,previous.write
RUN='20261011-phase1-stage2-rx-actions-manysig-m15-r01'
RELEASE='cvs_stage2_rx_actions_20261011_r01'
BASE=Path(PROJECT)/'runs'/RUN;OWNER='codex/root/stage2-rx-actions-20261011'
ARMS=('CE','SSL','PHYS','ROT','CRIT');PRIMARY='CRIT'
PAIRS=[('SSL','CE'),('PHYS','SSL'),('ROT','SSL'),('PHYS','ROT'),('CRIT','PHYS')]+[(a,'P0') for a in ARMS]
learning_rate=previous.learning_rate;frequency_learning_rate=previous.frequency_learning_rate

def rows():return [dict(arm=a,model_seed=s,row_id=f'{a}-s{s}') for s in SEEDS for a in ARMS]

def config(row):
 if row not in rows():raise ValueError('Unregistered RX stage2 row')
 base='O_CE' if row['arm']=='CE' else 'O_DIRECT';c=deepcopy(previous.config(next(r for r in previous.rows() if r['arm']==base and r['model_seed']==row['model_seed'])))
 c.update(schema='stage2_rx_actions_v1',run_id=RUN,**row,stage='phase1_stage2_rx_actions',method='cvs_stage2_rx_actions',method_name=row['arm'],output_root=(BASE/row['row_id']/'source').as_posix(),launch_owner=OWNER)
 c['losses'].update(soft_identity=.25,critic=1.)
 c['rx_action']=dict(enabled=row['arm'] in ('PHYS','ROT','CRIT'),critic=row['arm']=='CRIT',kind='rotated' if row['arm']=='ROT' else 'physical',taps=33,scales=[.5,1.],max_amplitude_dB=3.,start_after_updates=888,ramp_updates=888,L_augmented_CE_fraction=.5,critic_every=4,critic_batch=32,critic_lr=2e-4,critic_hidden=48,critic_uniform_warmup_updates=444,critic_uniform_probability=.5,critic_temperature=.25,L_fit='other five TX AND other two days',U_fit='all six L TX, other two days; U TX never read',critic_fit='actual deterministic CE increment on up to32L, identity detached; smoothL1; zero-action anchored',action_selection='eight candidates: four other sourceRX times two fixed scales; no identity class prediction in bank lookup')
 c['evaluation']=dict(primary_arm=PRIMARY,goal_three_seed_mean=.84,all_predeclared_arms_reported=True,truth_last=True,feedback_to_training=False,fixed_control_run=None,fixed_control_arms=[],control_role='same40epoch budget and E200 parents; five fixed arms, no sourceV ranking')
 c['mechanism_scope']='Original identity gates/reference retained. Physical bank comes from L IQ only. Auxiliary critic explains bounded action difficulty and selects identity training inputs; it never duplicates TX classification. No EMA/satellite.'
 return c

def validate(c):
 r=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if r is None or c!=config(r):raise ValueError('RX stage2 configuration differs')
 return c
