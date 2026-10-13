from copy import deepcopy
from pathlib import Path
from experiments.cvs_frequency_context50 import design as parent
from experiments.cvs_legacy_no_sat import design as legacy
LEGACY_FIELDS=parent.LEGACY_FIELDS
from experiments.cvs_default_head.predict import READOUTS
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,SEEDS,FULL_FP32_POLICY,read,write=(getattr(parent,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','SEEDS','FULL_FP32_POLICY','read','write'))
RUN='20261014-phase1-pa-cross-delay-legacycos-clean-manysig-m9-r01'
RELEASE='cvs_pa_cross_delay_20261014_r01'
OWNER='codex/root/pa-cross-delay-20261014'
BASE=Path(PROJECT)/'runs'/RUN
EPOCHS=50;LABEL_EPOCHS=32;PSEUDO_EPOCHS=18;STEPS=222;UPDATES=11100
ARMS=('LEGACY50','PA_CONV','PA_CROSS_DELAY');PRIMARY='PA_CROSS_DELAY'
VIEWS=('clean',);QUERY_COUNT=168000
BASELINE_RUN=parent.RUN
BASELINE_EVAL_RUN='20261013-phase1-frequency-context-eval-recovery-manysig-m9-r01'
PA_FIELDS=dict(rank=8,memory=9,output_channels=24,filter_normalization='row L2 eps1e-4',input_clip=2.,initial_gain=.05,activation='SiLU after local terms',projection='Conv1d16to24 k1',insertion='before existing PA EnvelopeGate1d',old_pa_preserved=True,classifier_change=False,per_packet_only=True,receiver_metadata=False)
def rows():return [dict(row_id=f'{a}-s{s}',arm=a,model_seed=s) for s in SEEDS for a in ARMS]
def training_rows():return [r for r in rows() if r['arm']!='LEGACY50']
def baseline_config(row):return parent.config(dict(row_id=f"LEGACY50-s{row['model_seed']}",arm='LEGACY50',model_seed=row['model_seed']))
def config(row):
 if row not in rows():raise ValueError('Unregistered PA row')
 old=baseline_config(row);c=deepcopy(old)
 c.update(schema='pa_cross_delay_legacy50_v1',run_id=RUN,**row,stage='evaluation_reuse' if row['arm']=='LEGACY50' else 'pa_cross_delay',method='cvs_pa_cross_delay',output_root=(BASE/row['row_id']/'source').as_posix(),launch_owner=OWNER,source_config=old if row['arm']=='LEGACY50' else None,model_initialization='frozen baseline reuse' if row['arm']=='LEGACY50' else 'scratch',source_V='health only; final50 fixed, no performance selection',pa_cross_delay=dict(PA_FIELDS,enabled=row['arm']!='LEGACY50',mode='cross_delay' if row['arm']=='PA_CROSS_DELAY' else 'conv' if row['arm']=='PA_CONV' else 'none'),readouts=list(READOUTS),classifier_training='default jointly with backbone; no separate fitting',goal=dict(primary=PRIMARY,clean_three_seed_mean=.84,readout='ORIGINAL_HEAD',feature_evidence='same default head; feature structure changes only; true newTX/Phase2 separate',feedback_to_training=False))
 return c
def validate(c):
 r=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if r is None or c!=config(r) or c['epochs']>50 or c['readouts']!=['ORIGINAL_HEAD']:raise ValueError('Fixed PA/default-head matrix differs')
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
