from copy import deepcopy
from pathlib import Path
from experiments.cvs_episodic_transfer import design as parent
from experiments.cvs_legacy_no_sat import design as legacy
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,SEEDS,FULL_FP32_POLICY,read,write=(getattr(parent,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','SEEDS','FULL_FP32_POLICY','read','write'))
RUN='20261013-phase1-cross-frequency-legacycos-clean-manysig-m9-r01'
RELEASE='cvs_cross_frequency_20261013_r01'
OWNER='codex/root/cross-frequency-20261013'
BASE=Path(PROJECT)/'runs'/RUN
EPOCHS=50;LABEL_EPOCHS=32;PSEUDO_EPOCHS=18;STEPS=222;UPDATES=11100
ARMS=('LEGACY50','POWER_RELATIONS','COMPLEX_RELATIONS');PRIMARY='COMPLEX_RELATIONS'
READOUTS=parent.READOUTS;VIEWS=('clean',);QUERY_COUNT=168000
LEGACY_FIELDS=parent.LEGACY_FIELDS
BASELINE_RUN=parent.BASELINE_RUN
BASELINE_EVAL_RUN=parent.BASELINE_EVAL_RUN
BASELINE_READOUT_RUN=parent.RUN
RELATION_FIELDS=dict(epsilon=1e-4,fft_size=256,pairs=['adjacent+1','mirror-k','distant+16'],relation_channels=6,position_channels=2,conv_channels=32,kernel_size=3,pool_positions=8,output_dimension=160,initial_gain=.01,classifier_change=False,per_packet_only=True,receiver_metadata=False)
def rows():return [dict(row_id=f'{a}-s{s}',arm=a,model_seed=s) for s in SEEDS for a in ARMS]
def training_rows():return [r for r in rows() if r['arm']!='LEGACY50']
def baseline_config(row):return parent.baseline_config(dict(row_id=f"LEGACY50-s{row['model_seed']}",arm='LEGACY50',model_seed=row['model_seed']))
def baseline_readout_config(row):return parent.config(dict(row_id=f"LEGACY50-s{row['model_seed']}",arm='LEGACY50',model_seed=row['model_seed']))
def config(row):
 if row not in rows():raise ValueError('Unregistered cross-frequency row')
 old=baseline_config(row);c=deepcopy(old)
 c.update(schema='cross_frequency_legacy50_v1',run_id=RUN,**row,stage='evaluation_reuse' if row['arm']=='LEGACY50' else 'cross_frequency',method='cvs_cross_frequency',output_root=(BASE/row['row_id']/'source').as_posix(),launch_owner=OWNER,source_config=old if row['arm']=='LEGACY50' else None,model_initialization='frozen baseline reuse' if row['arm']=='LEGACY50' else 'scratch',source_V='health only; final50 fixed, no performance selection',cross_frequency=dict(RELATION_FIELDS,enabled=row['arm']!='LEGACY50',mode='complex' if row['arm']=='COMPLEX_RELATIONS' else 'power' if row['arm']=='POWER_RELATIONS' else 'none'),readouts=list(READOUTS),fresh_cosine_probe=deepcopy(parent.config(parent.rows()[0])['fresh_cosine_probe']),goal=dict(primary=PRIMARY,clean_three_seed_mean=.84,readout='ORIGINAL_HEAD',feature_evidence='same ordinary fresh cosine/prototype; true newTX/Phase2 separate',feedback_to_training=False))
 return c
def validate(c):
 r=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if r is None or c!=config(r) or c['epochs']>50:raise ValueError('Fixed cross-frequency matrix differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c)
 if c['arm']=='LEGACY50':raise ValueError('Frozen baseline must not be retrained')
 a=legacy.parent.args_from_validated_config(c,device)
 for key,value in LEGACY_FIELDS.items():
  if getattr(a,key,None)!=value and key not in ('phase1_lr_schedule','phase1_lr_min'):raise ValueError('Historical recipe drift: '+key)
  setattr(a,key,value)
 a.epochs=50;a.label_epochs=32;a.pseudo_epochs=18;a.pseudo_temporal_bank_min_streak=2
 if a.use_mixstyle or a.use_muse_ssdg or a.os_gradient_surgery or a.os_budget_controller:raise ValueError('Unregistered training mechanism')
 return a
