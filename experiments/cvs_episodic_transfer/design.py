from copy import deepcopy
from pathlib import Path
from experiments.cvs_frequency_context50 import design as parent
from experiments.cvs_legacy_no_sat import design as legacy
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,SEEDS,FULL_FP32_POLICY,read,write=(getattr(parent,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','SEEDS','FULL_FP32_POLICY','read','write'))
RUN='20261013-phase1-episodic-feature-legacycos-clean-manysig-m9-r01'
RELEASE='cvs_episodic_feature_20261013_r01'
OWNER='codex/root/episodic-feature-20261013'
BASE=Path(PROJECT)/'runs'/RUN
EPOCHS=50;LABEL_EPOCHS=32;PSEUDO_EPOCHS=18;STEPS=222;UPDATES=11100
ARMS=('LEGACY50','EPISODE_SAME_RX','EPISODE_CROSS_RX');PRIMARY='EPISODE_CROSS_RX'
READOUTS=('ORIGINAL_HEAD','FRESH_COSINE_ALL_L','COSINE_PROTO_ALL_L');VIEWS=('clean',);QUERY_COUNT=168000
SOURCE_RX=(1,3,4,6,8);INTERVAL=4;EPISODES=UPDATES//INTERVAL;AUX_WEIGHT=.2
LEGACY_FIELDS=parent.LEGACY_FIELDS
BASELINE_RUN=parent.RUN
BASELINE_EVAL_RUN='20261013-phase1-frequency-context-eval-recovery-manysig-m9-r01'
def rows():return [dict(row_id=f'{a}-s{s}',arm=a,model_seed=s) for s in SEEDS for a in ARMS]
def training_rows():return [r for r in rows() if r['arm']!='LEGACY50']
def baseline_config(row):return parent.config(next(r for r in parent.rows() if r['arm']=='LEGACY50' and r['model_seed']==row['model_seed']))
def config(row):
 if row not in rows():raise ValueError('Unregistered episodic feature row')
 old=baseline_config(row);c=deepcopy(old)
 c.update(schema='episodic_feature_legacy50_v1',run_id=RUN,**row,stage='evaluation_reuse' if row['arm']=='LEGACY50' else 'episodic_feature',method='cvs_episodic_transfer',output_root=(BASE/row['row_id']/'source').as_posix(),launch_owner=OWNER,source_config=old if row['arm']=='LEGACY50' else None,model_initialization='frozen baseline reuse' if row['arm']=='LEGACY50' else 'scratch',source_V='health only; final50 fixed, no performance selection',source_episode=dict(enabled=row['arm']!='LEGACY50',mode='cross_rx' if row['arm']=='EPISODE_CROSS_RX' else 'same_rx' if row['arm']=='EPISODE_SAME_RX' else 'none',interval=INTERVAL,start_step=INTERVAL,weight=AUX_WEIGHT,support_per_class=2,query_per_class=4,source_rx=list(SOURCE_RX),query_rx_rule='round robin all source RX',support_rx_rule='next two ring RX, one each/class' if row['arm']=='EPISODE_CROSS_RX' else 'same query RX, disjoint samples',ridge=.1,logit_scale=10.,normalization_epsilon=1e-4,feature_boundary='actual joint160, no class W',L_only=True,U_truth_hidden=True,V_training=False,main_rng_preserved=True),readouts=list(READOUTS),fresh_cosine_probe=dict(scale=30.,bias=False,epsilon=1e-4,epochs=20,optimizer='Adam',lr=.01,weight_decay=0.,batch_size=256,initialization='seeded Xavier model_seed+60013; no old W'),goal=dict(primary=PRIMARY,clean_three_seed_mean=.84,readout='ORIGINAL_HEAD',feature_evidence='same ordinary fresh cosine/prototype and source probes; true newTX/Phase2 separate',feedback_to_training=False))
 return c
def validate(c):
 r=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if r is None or c!=config(r) or c['epochs']>50:raise ValueError('Fixed episodic feature matrix differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c)
 if c['arm']=='LEGACY50':raise ValueError('Frozen baseline must not be retrained')
 a=legacy.parent.args_from_validated_config(c,device)
 for key,value in LEGACY_FIELDS.items():
  if getattr(a,key,None)!=value and key not in ('phase1_lr_schedule','phase1_lr_min'):raise ValueError('Historical recipe drift: '+key)
  setattr(a,key,value)
 a.epochs=50;a.label_epochs=32;a.pseudo_epochs=18;a.pseudo_temporal_bank_min_streak=2
 if a.use_mixstyle or a.use_muse_ssdg or a.os_gradient_surgery or a.os_budget_controller:raise ValueError('Unregistered gradient/control interaction')
 return a
