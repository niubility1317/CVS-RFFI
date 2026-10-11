from copy import deepcopy
from pathlib import Path
import math
from experiments.cvs_stage2_geometry import design as g

ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES=g.PROJECT,g.SOURCE,g.CAPSULE,g.TRUTH,g.CLASSES
SEEDS=g.SEEDS
FULL_FP32_POLICY=g.FULL_FP32_POLICY
read,write=g.read,g.write
parent_config=g.parent_config
AUDIT_RUN,AUDIT_BASE=g.AUDIT_RUN,g.AUDIT_BASE
RUN='20261011-phase1-stage2-hybrid-manysig-m33-r01'
RELEASE='cvs_stage2_hybrid_20261011_r01'
BASE=Path(PROJECT)/'runs'/RUN
RELEASE_BASE=Path(PROJECT)/'releases'/RELEASE
OWNER='codex/root/stage2-hybrid-20261011'
ARMS=tuple('H'+str(i) for i in range(11))
NAMES=dict(H0='CE',H1='CE + cross-RX supervised contrastive',H2='CE + same-TX cross-RX feature mix',
 H3='CE + between-TX manifold mix',H4='CE + cross-RX contrastive + cross-RX feature mix',
 H5='SAM CE',H6='CE + cyclic-tail weight averaging',H7='SAM + cyclic-tail weight averaging',
 H8='H4 + coverage finite L/T',H9='H4 + SAM + cyclic-tail averaging + coverage finite L/T',
 H10='H4 + balanced conditional finite L/T')
PAIRS=[(a,'H0') for a in ARMS if a!='H0']+[('H4','H1'),('H4','H2'),('H7','H5'),('H8','H4'),('H9','H8'),('H10','H8')]


def rows():return [dict(arm=arm,model_seed=seed,row_id=f'{arm}-s{seed}') for seed in SEEDS for arm in ARMS]


def config(row):
 p=parent_config(row['model_seed']);arm=row['arm']
 return dict(schema='stage2_hybrid_v1',run_id=RUN,**row,method='cvs_stage2_hybrid',variant='reference_response',
  method_name=NAMES[arm],dataset=p['dataset'],source_contract=SOURCE,parent_checkpoint=p['parent_checkpoint'],
  parent_prediction=p['parent_prediction'],checkpoint_sources=p['checkpoint_sources'],parent_contract=p,
  preparation=(AUDIT_BASE/f"s{row['model_seed']}").as_posix(),output_root=(BASE/row['row_id']/'source').as_posix(),
  epochs=40,steps_per_epoch=222,updates=8880,batch_size=128,label_smoothing=0.,optimizer=deepcopy(p['optimizer']),
  enabled=dict(contrastive=arm in ('H1','H4','H8','H9','H10'),cross_rx_mix=arm in ('H2','H4','H8','H9','H10'),
   manifold_mix=arm=='H3',sam=arm in ('H5','H7','H9'),averaging=arm in ('H6','H7','H9'),
   actions=arm in ('H8','H9','H10'),conditional_actions=arm=='H10'),
  losses=dict(contrastive=.05,cross_rx_mix=.25,manifold_mix=.25,finite_lt=1.,keep=.1,ramp_steps=444,
   margin_required=math.log(95.),temperature=.1),
  mixing=dict(beta=.4,cross_rx_lambda='Uniform[.25,.75]',normalized_joint=True,head_scale=30.),
  sam=dict(rho=.02,adaptive=False,same_batch_same_global_RNG=True,base_optimizer='AdamW',temporary_perturbation_is_not_update=True),
  averaging=dict(start_epoch=31,cycle_epochs=2,cycle_peak_fraction=.5,snapshot_epochs=[32,34,36,38,40],
   buffers='fixed final buffers; BatchNorm architecture forbidden',selection='fixed five-snapshot mean; no V choice'),
  checkpoint_selection='fixed_tail_average' if arm in ('H6','H7','H9') else 'final_only',
  action=dict(policy='balanced_conditional' if arm=='H10' else 'coverage',payload=(RELEASE_BASE/'actions'/f"s{row['model_seed']}.pt").as_posix() if arm=='H10' else None,
   fit='fresh original E200, same-scale tanh coordinates, abs noise, first full pass raw margin then full pass finite risk',
   every_steps=4,batch=32,order=['linear','linear','temporal','temporal'],coverage_fraction=.5),
  sampling='Same ordinary permutation and relation replacement streams as B0; no target/sample selection',
  target_access=False,unlabeled=False,teacher=False,satellite=False,test_views=['clean'],
  numerical_policy=deepcopy(FULL_FP32_POLICY),concurrency=dict(max_processes_per_gpu=3,includes_existing=True,gpu_ids=list(range(8))),
  evaluation=dict(primary_arm='H9',goal_three_seed_mean=.84,all_predeclared_arms_reported=True,truth_last=True,feedback_to_training=False),launch_owner=OWNER)


def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Unregistered hybrid matrix/config')
 return c


def learning_rate(c,step,peak):
 from experiments.cvs_stage2_decouple.common import learning_rate as original
 if not c['enabled']['averaging'] or step<=30*222:return original(step,peak)
 local=(step-30*222-1)%(2*222)
 return 1e-6+.5*(.5*peak-1e-6)*(1+math.cos(math.pi*local/(2*222-1)))
