"""Fixed dual-view teachers and feature-only temporal adapters; E20+10."""
import json
from copy import deepcopy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN='20261019-phase1-F-dualTeacher-adapter-e20plus10-s2026092701-m6-r02'
RELEASE='cvs_F_dualTeacher_adapter_E30_20261019_r02'
OWNER='codex/root/F-dualTeacherAdapter30'
SEED=2026092701;STEPS=222;EPOCHS=10;UPDATES=2220;PARENT_EPOCH=20;PARENT_UPDATES=4440
PROJECT=json.loads((ROOT/'runtime.json').read_text(encoding='utf-8-sig'))['project'] if (ROOT/'runtime.json').is_file() else 'PROJECT'
BASE=Path(PROJECT)/'runs'/RUN
PARENT_RUN='20261019-phase1-F-nontrue-CSIL-e20-s2026092701-m4-r01'
PARENT_ROW='B0-s2026092701';PARENT_SOURCE=Path(PROJECT)/'runs'/PARENT_RUN/PARENT_ROW/'source'
REFERENCE_RELEASE='cvs_F_epochReplay_E20_20261019_r01'
REFERENCE_COMMIT='39db239b9f7ac479a84080ff1b0f5ae363feb76c'
TEACHER_RUN='20260930-phase1-native-baselines-practical-manysig-m5-r01'
TEACHER_ROW='csil-s2026092701'
TEACHER_SOURCE=Path(PROJECT)/'paper_reproduction/runs'/TEACHER_RUN/TEACHER_ROW
FIT=(1,3,4,6,8);CLASSES=['14-10','14-7','20-15','20-19','6-15','8-20']
SCENES=('practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
VIEWS=('clean',*SCENES)
ARMS=('NOAS_FT','NOAS_ADAPTER','S0','S1','S2','S3')
SOURCE=Path(PROJECT)/'runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json'
FULL_FP32_POLICY=dict(cudnn_allow_tf32=False,cuda_matmul_allow_tf32=False,cudnn_benchmark=False,cudnn_deterministic=False,matmul_precision='highest')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def rows():return [dict(arm=a,row_id=a+'-s'+str(SEED),model_seed=SEED) for a in ARMS]
def parent_config():
 from experiments.cvs_F_nontrue20 import design as old
 return old.config(next(r for r in old.rows() if r['arm']=='B0'))
def model_config():
 from experiments.cvs_F_nontrue20 import design as old
 return old.model_config()
def config(row):
 if row not in rows():raise ValueError('Unregistered one-seed dual-teacher matrix row')
 a=row['arm'];adapter=a not in ('NOAS_FT','S0');As=a not in ('NOAS_FT','NOAS_ADAPTER')
 return dict(schema='F_dualTeacher_adapter30_v1',run_id=RUN,**row,launch_owner=OWNER,output_root=str(BASE/row['row_id']/'source'),project=PROJECT,
  epochs=10,label_epochs=0,pseudo_epochs=10,steps_per_epoch=222,updates=2220,parent_epochs=20,parent_updates=4440,cumulative_epochs=30,cumulative_updates=6660,
  parent_source=str(PARENT_SOURCE),checkpoint_sources=[dict(run_id=PARENT_RUN,row_id=PARENT_ROW,checkpoint=str(PARENT_SOURCE/'final_ssdg.pth'),selection='fixed FINAL20',scope='student and own parent EMA')],
  initialization='strict own cleanF B0 FINAL20, then zero-output adapter when applicable',head='unchanged default6x160 cosine scale30; all head parameters frozen in every row',
  features='SIG_TIME_X3+CENTER_25K; NoExplicitCFO; original physical paths preserved',
  adapter=dict(enabled=adapter,rank=16,kernel=5,bias=False,activation='GELU',positions=['t1_pre_norm','t2_pre_norm'],zero_B=True,zero_gate=False,init_seed=SEED,parameters=4768 if adapter else 0),
  original_F_trainable=not adapter,default_head_trainable=False,first_view='ordinary' if a=='NOAS_FT' else 'raw',As=As,
  loss=dict(clean_CE_weight=.5 if As else 1.,satellite_CE_weight=.5 if As else 0.,label_smoothing=.01,CE_temperature=1.,
   clean_KD_weight=.25 if a in ('NOAS_ADAPTER','S2','S3') else 0.,satellite_KD_weight=.25 if a=='S3' else 0.,
   clean_temperature=4.,satellite_temperature=4.,clean_teacher_scale=1.,clean_student_scale=1.,satellite_teacher_scale=6.,satellite_student_scale=1.,
   denominator='all128 for each CE and masked KL',teacher_correct_mask=True,student_gate=False,temperature_compensation=True),
  teachers=dict(clean=dict(run_id=PARENT_RUN,row_id=PARENT_ROW,epochs=20,selection='fixed FINAL20'),satellite=dict(run_id=TEACHER_RUN,row_id=TEACHER_ROW,epochs=200,updates=9800,selection='fixed last200'),
   both_forwards_every_row=True,teacher_packets_per_L_step_each=128,shadow_when_loss_zero=True,register_in_student=False,new_teacher_training_epochs=0,
   class_order=CLASSES,logit_units=dict(common_cosine_scale=30,clean_native_scale=30,satellite_native_scale=5,satellite_native_bias=5,source_fitted_temperature=False)),
  optimizer=dict(name='AdamW',reset=True,lr=.0002,lr_min=.000001,schedule='newcosine10;notcontinuouscos30',weight_decay=.0001,clip=5.,
   Sinc='frozen0updates' if adapter else 'CENTER_UNITS qgroup LR1000/eps1000inv;physical decay/clip;head remains frozen'),
  ownEMA=dict(parent='own B0 parent ema_model strict original keys plus identical zero-output adapter keys',decay=.999,startup_average=False,local_update_start=0,local_final_updates=2220,cumulative_final_updates=6660),
  U=dict(recipe='original hiddenTX batch_neighbor/strong/confidence/ownEMA;no new U hypothesis',TX_hidden=True,student_packets=567000,EMA_packets=567000,cache='no crossbatch epoch bank inherited'),
  sampler=dict(reset=True,rule='same fixed role-specific Generator on DataLoader and RandomSampler in every new segment',actual_sequences_logged=True),
  channel=dict(input='rawL',full_practical=True,fresh_every_step=True,cumulative_epoch_range=[21,30],cumulative_step_range=[4441,6660],namespace='FcrossRX/train/e{cumulative_epoch}/step{cumulative_step}/L_S',NoAs='same channel/teacher shadow only; no student As CE'),
  data=dict(L=6300,U=56700,V=27000,source_RX=list(FIT),source_days=[1,2,3],split=392005,unchanged_contract=True),
  new_student_NN=dict(L=568320 if As else 284160,U=567000,ownEMA=567000,clean_teacher=284160,satellite_teacher=284160),
  source_promotion=dict(primary='S3',midlow_S3_minus_S2_min_pp=.5,fullVclean_S3_minus_max_NOAS_FT_NOAS_ADAPTER_min_pp=-.2,decision_before_formal_scores=True,automatic_extension=False),
  test_views=list(VIEWS),checkpoint_selection='fixed FINAL30;no source best or target selection',target_access=False,head_fit_updates=0,new_pseudo_mechanism=False,
  scientific_scope='Two frozen view-specific teachers plus new shallow/mid temporal feature modules;not closed epochReplay/nontrue rescue',
  baseline_note='NOAS_FT preserves original ordinary NoAs continuation;NOAS_ADAPTER matches raw+cleanKD adapter.NoAs cannot be weakened to manufacture gain;reportboth and parentB0.',
  budget_authorization='User latest dual-teacher/adapter report;explicit E20+10 cumulativeE30,one seed;historicalTc20/Ts200 costs separate')
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed parent/adapter/teacher/head/recipe/budget differs')
 return c

GPU_IDS=(4,6,7)
FORMAL_VIEWS=Path(PROJECT)/'runs/20261002-phase1-cvs-residual-sixscene-manysig-m8-r01/received_views'
TRUTH=Path(PROJECT)/'runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json'
PANEL=Path(PROJECT)/'runs/20261017-diagnostic-FcrossRX-fullsource-residual-s2026092701-m3-r01/diagnostic'
SOURCE_CACHE=Path(PROJECT)/'runs/20261011-phase1-stage2-geometry-source-manysig-m3-r01/s2026092701'
def make_args(c,device='cuda:0'):
 from .source import make_args as native_args
 return native_args(c,device)
