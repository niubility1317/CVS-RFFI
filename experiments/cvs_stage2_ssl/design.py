from pathlib import Path
from copy import deepcopy
from experiments.cvs_stage2_hybrid import design as h

ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES=h.PROJECT,h.SOURCE,h.CAPSULE,h.TRUTH,h.CLASSES
SEEDS=h.SEEDS
read,write=h.read,h.write
FULL_FP32_POLICY=h.FULL_FP32_POLICY
AUDIT_RUN,AUDIT_BASE=h.AUDIT_RUN,h.AUDIT_BASE
parent_config=h.parent_config
RUN='20261011-phase1-stage2-ssl-hybrid-manysig-m48-r01'
RELEASE='cvs_stage2_ssl_hybrid_20261011_r01'
BASE=Path(PROJECT)/'runs'/RUN
RELEASE_BASE=Path(PROJECT)/'releases'/RELEASE
OWNER='codex/root/stage2-ssl-hybrid-20261011'
ARMS=tuple(f'S{i}' for i in range(16))
BASE_ARMS=dict(S0='H0',S1='H0',S2='H1',S3='H2',S4='H3',S5='H4',S6='H5',S7='H7',S8='H8',S9='H9',S10='H10',S11='H4')
BASE_ARMS.update(S12='H0',S13='H4',S14='H9',S15='H4')
NAMES={arm:('Legacy batch-neighbor SSL' if arm=='S0' else 'Visit-pass SSL')+' + '+h.NAMES[BASE_ARMS[arm]]+(' + accepted-U cross-RX contrastive' if arm=='S11' else '') for arm in ARMS}
PAIRS=[('S0','H0'),('S1','S0')]+[(a,'S1') for a in ARMS[2:]]+[('S5','S2'),('S5','S3'),('S7','S6'),('S8','S5'),('S9','S8'),('S10','S8'),('S11','S5')]
PAIRS += [('S12','S1'),('S13','S5'),('S14','S9'),('S15','S13'),('S13','S12')]
UNLABELED_CACHE=RELEASE_BASE/'source_unlabeled/source_cache.pt'
CONTROL_RUN=h.RUN
CONTROL_ARMS=('H0',)


def rows():return [dict(arm=arm,model_seed=s,row_id=f'{arm}-s{s}') for s in SEEDS for arm in ARMS]


def config(row):
 arm=row['arm'];parent=h.config(dict(arm=BASE_ARMS[arm],model_seed=row['model_seed'],row_id=f"{BASE_ARMS[arm]}-s{row['model_seed']}"))
 c=deepcopy(parent)
 c.update(schema='stage2_ssl_hybrid_v1',run_id=RUN,**row,method='cvs_stage2_ssl_hybrid',method_name=NAMES[arm],
  output_root=(BASE/row['row_id']/'source').as_posix(),teacher=True,unlabeled=True,launch_owner=OWNER,
  semi=dict(cache=UNLABELED_CACHE.as_posix(),batch_size=256,role='U_s',count=56700,mode='batch_neighbor' if arm=='S0' else 'visit_pass_bank',
   teacher='clone matched E200 identity; eval/no_grad; cache-safe EMA after one successful optimizer update',ema_decay=.999,
   pseudo_weight=.16,entropy_weight=.01,ramp_steps=444,tau_min=.92,tau_max=.97,quantile=.86,temporal_window=2,
   temporal_min_conf=.80,min_streak=2,strong_noise_std=.015,strong_agreement=True,domain_gate=False,
   warmup='No extra 130+70 scratch stages: mature original E200; all40 epochs enable semi with444-step ramp',
   feature_contrastive_use_accepted_U=arm=='S11',truth_access=False,history_updates_per_optimizer_step=1,
   teacher_selected_for_test=False,SAM_plan='teacher/history/strong input fixed once per update; reused for both closures'),
  evaluation=dict(primary_arm='S13',goal_three_seed_mean=.84,all_predeclared_arms_reported=True,truth_last=True,feedback_to_training=False,
   fixed_control_run=CONTROL_RUN,fixed_control_arms=list(CONTROL_ARMS),control_role='H0 fixed clean predictions only; never E40 initialization'))
 if arm in ('S12','S13','S14','S15'):
  c['optimizer']['encoder_peak_lr']=.0003 if arm=='S15' else .0001
  c['optimizer']['fusion_head_peak_lr']=.001 if arm=='S15' else .0003
  c['method_name'] += ' + stronger full-identity optimization '+('10x' if arm=='S15' else 'approximately3x')
 c['optimization_scope']='all native identity encoder/fusion/classifier parameters; no frozen identity layers; domain networks frozen'
 return c


def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Unregistered SSL continuation matrix')
 return c


learning_rate=h.learning_rate
