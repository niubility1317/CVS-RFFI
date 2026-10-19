"""Local fixed 2x2 feature-learning ablation; all choices precede target access."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2]
RUN='20261019-phase1-FcrossRX-TXgeometry-e20-local-m4-r02'
SEEDS=(2026092701,2026092702,2026092703)
ARMS=('BASE','WEIGHT','GEOMETRY','BOTH')
VIEWS=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
EPOCHS=20;STEPS=222;UPDATES=EPOCHS*STEPS
CLASSES=('14-10','14-7','20-15','20-19','6-15','8-20')
CALIBRATION=ROOT/'automation_reports/CV-SincNet/20261019-diagnostic-FcrossRX-TXrisk-sourceV-m3-r01/source_calibration.json'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def rows(seeds=SEEDS):return [dict(row_id=f'{a}-s{s}',arm=a,model_seed=s) for s in seeds for a in ARMS]
def config(row):
 if row not in rows():raise ValueError('Registered factorial rows only')
 cal=read(CALIBRATION)
 if cal['class_names']!=list(CLASSES) or cal['calibration_role']!='sourceV only' or cal['target_feedback']:raise ValueError('Source-only class calibration required')
 return dict(schema='local_F_TXgeometry20_v1',run_id=RUN,**row,epochs=EPOCHS,steps_per_epoch=STEPS,optimizer_steps=UPDATES,batch_size=128,
  weights=cal['weights'] if row['arm'] in ('WEIGHT','BOTH') else [1.]*6,geometry=row['arm'] in ('GEOMETRY','BOTH'),
  shallow_middle_lambda=.1,joint160_lambda=.05,temperature=.2,warm_epochs=5,ramp_epochs=10,
  optimizer='CENTER_UNITS AdamW physicalHz clip5',lr=.0002,lr_min=.000001,weight_decay=.0001,
  label_smoothing=.01,RDrop_alpha=1.,satellite_CE_weight=.68,satellite_consistency_weight=0.,
  ordinary_augmentation=True,satellite_concat=True,satellite_start=1,satellite_seed=2027,
  source_split_seed=392005,source_roles=dict(L_s=6300,U_s=56700,V=27000),U_used=False,EMA=False,
  checkpoint_selection='fixed_final20',initialization='scratch',checkpoint_sources=[],default_head_joint=True,separate_head_training=False,
  test_views=list(VIEWS),evaluation_seed=392005,target_feedback=False)
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Frozen source/target/defaulthead recipe differs')
 return c

def expansion(source):
 """Decide on all firstseed source rows before loading any target observations."""
 if set(source)!=set(ARMS):raise ValueError('Complete firstseed source factorial required')
 import numpy as np
 base=source['BASE'];weak=np.argsort(read(CALIBRATION)['risk'])[-2:];qualified=[];detail={}
 for arm in ARMS[1:]:
  s=source[arm];weak_gain=sum(s['recall'][k]-base['recall'][k] for k in weak)/2
  overall=s['accuracy']-base['accuracy'];worst=s['worst_RX']-base['worst_RX'];class_drop=min(a-b for a,b in zip(s['recall'],base['recall']))
  passed=weak_gain>=.002 and overall>=-.002 and worst>=-.005 and class_drop>=-.01
  detail[arm]=dict(PASS=bool(passed),weak_TX_gain_pp=100*weak_gain,overall_difference_pp=100*overall,worst_RX_difference_pp=100*worst,minimum_class_difference_pp=100*class_drop)
  if passed:qualified.append(arm)
 return dict(status='SOURCE_EXPANSION_FIXED',weak_indices=weak.tolist(),qualified=qualified,expand_arms=['BASE',*qualified] if qualified else [],remaining_seeds=list(SEEDS[1:]),criterion='weakTX mean +0.2pp; overall >=-0.2pp; worstRX >=-0.5pp; anyTX >=-1pp; all relative to matched BASE; no target feedback',detail=detail,target_access=False)
