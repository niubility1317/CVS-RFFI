from copy import deepcopy
from pathlib import Path
from experiments.cvs_legacy_no_sat import design as legacy
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,SEEDS,FULL_FP32_POLICY,read,write=(getattr(legacy,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','SEEDS','FULL_FP32_POLICY','read','write'))
RUN='20261013-phase1-frequency-context-legacycos-clean-manysig-m9-r01'
RELEASE='cvs_frequency_context50_20261013_r01'
OWNER='codex/root/frequency-context-legacycos-20261013'
BASE=Path(PROJECT)/'runs'/RUN
EPOCHS=50;LABEL_EPOCHS=32;PSEUDO_EPOCHS=18;STEPS=222;UPDATES=EPOCHS*STEPS
ARMS=('LEGACY50','LOCAL_FREQ','GLOBAL_FREQ');PRIMARY='GLOBAL_FREQ';VIEWS=('clean',)
QUERY_COUNT=168000
LEGACY_FIELDS=dict(use_unlabeled=True,use_ema_teacher=True,ema_decay=.999,a1_ema_versioned_updates=True,a1_ema_startup_average=False,pseudo_threshold_mode='rx_day_quantile',tau_min=.92,tau_max=.97,pseudo_quantile=.86,pseudo_domain_gate=False,pseudo_temporal_gate=True,pseudo_temporal_mode='batch_neighbor',pseudo_temporal_window=2,pseudo_temporal_min_conf=.8,pseudo_strong_agreement=True,strong_noise_std=.015,lambda_u=.16,lambda_ent=.01,label_smoothing=.01,use_aug=True,weight_decay=.0001,use_concat_sat_channel_aug=False,phase1_lr_schedule='cosine',phase1_lr_min=1e-6)
def rows():return [dict(row_id=f'{a}-s{s}',arm=a,model_seed=s) for s in SEEDS for a in ARMS]
def config(row):
 if row not in rows():raise ValueError('Unregistered frequency row')
 c=deepcopy(legacy.config(next(r for r in legacy.rows() if r['model_seed']==row['model_seed'])))
 c.update(schema='frequency_context_legacy50_v1',run_id=RUN,**row,stage='frequency_context',method='cvs_frequency_context50',output_root=(BASE/row['row_id']/'source').as_posix(),epochs=EPOCHS,steps_per_epoch=STEPS,updates=UPDATES,label_epochs=LABEL_EPOCHS,pseudo_epochs=PSEUDO_EPOCHS,launch_owner=OWNER,checkpoint_selection='final_only',test_views=list(VIEWS),explicit_CFO_identity_input=False,training_target_access=False,baseline='historical batch_neighbor pseudo gates + EMA + cosine; no satellite',legacy_recipe=dict(LEGACY_FIELDS),source_V='health only; final checkpoint fixed at50, no source performance selection',frequency_context=dict(enabled=row['arm']!='LEGACY50',levels=['f1','f2'] if row['arm']!='LEGACY50' else [],channels=[16,32],positions=[16,8],mode='global' if row['arm']=='GLOBAL_FREQ' else 'local' if row['arm']=='LOCAL_FREQ' else 'none',local_radius=1,attention_heads=4,query_key_value_width_multiplier=2,ffn_width_multiplier=4,residual_scale=.01,position='fixed sine/cosine frequency-band index; no receiver/TX metadata',classifier_change=False,per_packet_only=True),goal=dict(primary=PRIMARY,clean_three_seed_mean=.84,feedback_to_training=False))
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row) or not 1<=c['epochs']<=50 or c['updates']!=c['epochs']*c['steps_per_epoch'] or c['label_epochs']+c['pseudo_epochs']!=c['epochs']:raise ValueError('Fixed frequency/legacy recipe differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c);a=legacy.parent.args_from_validated_config(c,device)
 for key,value in LEGACY_FIELDS.items():
  if getattr(a,key,None)!=value and key not in ('phase1_lr_schedule','phase1_lr_min'):raise ValueError('Historical recipe drift: '+key)
  setattr(a,key,value)
 a.epochs=c['epochs'];a.label_epochs=c['label_epochs'];a.pseudo_epochs=c['pseudo_epochs'];a.pseudo_temporal_bank_min_streak=2
 return a
