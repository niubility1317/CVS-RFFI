from pathlib import Path
from copy import deepcopy
from experiments.cvs_F_nontrue20 import design as old
read,write=old.read,old.write
ROOT=Path(__file__).resolve().parents[2]
RUN='20261019-phase1-F-featureAdapters-e20plus10-s2026092701-m6-r02'
RELEASE='cvs_F_featureAdapters_E30_20261019_r02';LOCAL='local_artifacts/F_featureAdapters_E30_20261019_r02';OWNER='codex/root/F-feature-adapters'
PROJECT=read(ROOT/'runtime.json')['project'] if (ROOT/'runtime.json').is_file() else 'PROJECT'
BASE=Path(PROJECT)/'runs'/RUN
SEED=2026092701;RANK=8;KERNEL=5;ALPHA=8.;EPOCHS=10;STEPS=222;UPDATES=2220;BATCH=128
REFERENCE_RUN=old.RUN;REFERENCE_RELEASE='cvs_F_nontrue_CSIL_E20_20261019_r01';REFERENCE_COMMIT='1e7cc414f73f093dfbdccee0e4ba7cf90c7a5324'
SOURCE,FORMAL_VIEWS,TRUTH,PANEL=old.SOURCE,old.FORMAL_VIEWS,old.TRUTH,old.PANEL
FIT,SCENES,VIEWS,CLASSES,FULL_FP32_POLICY=old.FIT,old.SCENES,old.VIEWS,old.CLASSES,old.FULL_FP32_POLICY
PARENT_ROW='B0-s2026092701';HEAD='id_backbone.encoder.core.id_backbone.cls_head.weight'

def parent_config():return old.config(next(r for r in old.rows() if r['arm']=='B0'))

def rows():return [dict(row_id=style+'_'+name+'-s'+str(SEED),style=style,As=enabled,arm=style+'_'+name,model_seed=SEED) for style in ('FULL','LORA','TEMP') for name,enabled in (('NoAs',False),('As',True))]

def config(row):
 if row not in rows():raise ValueError('Unregistered fixed six-row feature-adaptation matrix')
 return dict(schema='F_feature_adapters30_v1',run_id=RUN,**row,launch_owner=OWNER,output_root=str(BASE/row['row_id']/'source'),parent=dict(run=REFERENCE_RUN,row=PARENT_ROW,checkpoint='source/final_ssdg.pth',selection='strict own source scratch FINAL20, never target selected',epochs=20,identity='pureF SIG_TIME_X3+CENTER_25K+CENTER_UNITS; no CROSS_RX/RDrop'),epochs=10,steps_per_epoch=222,updates=2220,cumulative_epochs=30,cumulative_backbone_exposure='shared parent20 + own supervised feature-adaptation10; original F frozen in LORA/TEMP, never free parent cost',source=dict(L=6300,U=56700,V=27000,RX=list(FIT),days=[1,2,3],split=392005,U_TX_hidden=True,adaptation_U_used=False),training=dict(model_mode='eval with autograd; common deterministic dropout-off objective, not native unchanged continuation',input='raw clean source IQ',source_sampler='fixed private shuffled full L cycling;222x128 per epoch;exact shared complete As stream',loss='CE0(raw)+CE0(second)+mean_all128[parentCorrect*KL(parent_raw||student_raw)]',second='complete practical(raw)' if row['As'] else 'same raw clean IQ, same total CE mass and two student forwards',label_smoothing=0.,As_weight=1.,keep_weight=1.,keep_temperature=1.,keep_only_parent_correct_L=True,student_As_confidence_gate=False,reference_parent_frozen=True,default_head_frozen=True,U_used=False,EMA_used=False),feature_adapter=dict(style=row['style'],layers=['t1','t2'],rank=RANK,kernel=KERNEL,lora_alpha=ALPHA,temporal='B SiLU(K_depthwise_rank(A block_input)), add after native pw before GN/activation/pool',init='A/K normal;B zero;no zero gate',CRRA_whitening=False,CRRA_conditioning=False,CRRA_support_gate=False,enabled_from_update=1,raw_IQ_bypass=False),optimizer=dict(type='reset AdamW',lr=.0002,min_lr=1e-6,weight_decay=.0001,betas=[.9,.999],eps=1e-8,schedule='cosine over new10 epochs, not continuous30',physical_clip=5.,Sinc='existing CENTER_UNITS algebra for FULL; unchanged frozen Sinc in LORA/TEMP'),checkpoint_selection='fixed FINAL30',test_views=list(VIEWS),new_classification_head=False,target_feedback=False)

def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Frozen parent/source/head/update-space/E20+10 matrix differs')
 return c
