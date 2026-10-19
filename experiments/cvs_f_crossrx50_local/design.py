"""Correct original50/no-satellite matched local weak-TX factorial."""
from copy import deepcopy
from pathlib import Path
import json
from . import baseline
from experiments.cvs_f_supcon50 import design as original
ROOT=Path(__file__).resolve().parents[2]
RUN='20261019-phase1-FcrossRX-TXrisk-geometry-e50-clean-local-m4-r01'
OWNER='codex/root/FcrossRX50-weakTX-local-20261019'
SEEDS=baseline.SEEDS;ARMS=('BASE','WEIGHT','GEOMETRY','BOTH');CLASSES=original.CLASSES
EPOCHS=50;LABEL_EPOCHS=33;PSEUDO_EPOCHS=17;STEPS=222;UPDATES=11100;VIEWS=('clean',)
FULL_FP32_POLICY=original.FULL_FP32_POLICY
CALIBRATION=ROOT/'automation_reports/CV-SincNet/20261019-diagnostic-FcrossRX-TXrisk-sourceV-m3-r01/source_calibration.json'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def rows(seeds=SEEDS):return [dict(row_id=f'{a}-s{s}',treatment=a,model_seed=s) for s in seeds for a in ARMS]
def config(row,base,profile,metadata):
 if row not in rows() or Path(base).name!=RUN:raise ValueError('Registered local original50 rows/output only')
 c=deepcopy(baseline.original_config(row['model_seed']));cal=read(CALIBRATION)
 if cal['class_names']!=list(CLASSES) or cal['calibration_role']!='sourceV only' or cal['target_feedback']:raise ValueError('Only prior original-F sourceV risk calibration')
 c.update(schema='F_crossRX50_TXgeometry_local_v1',run_id=RUN,row_id=row['row_id'],treatment=row['treatment'],launch_owner=OWNER,output_root=str(Path(base)/row['row_id']/'source'),runtime_base=str(base),runtime_profile=str(profile),runtime_metadata=str(metadata),baseline_reuse=False,
  class_weights=cal['weights'] if row['treatment'] in ('WEIGHT','BOTH') else [1.]*6,joint160_geometry=row['treatment'] in ('GEOMETRY','BOTH'),joint160_max_weight=.05,
  baseline_reference=dict(run_id=baseline.REFERENCE_RUN,commit=baseline.REFERENCE_COMMIT,arm='CROSS_RX',clean_mean_percent=85.11388888888888,role='Original method anchor; not inherited state. Local BASE freshly matched for runtime/cache migration'),
  training_recipe='Original F CROSS_RX50=33L+17U oldgate-ownEMA-cos RDrop1 LS.01 ordinaryAug;NO satellite;defaultheadjoint; declared weakTX feature treatment only',
  source_falsification=None,source_selection='Fixedfinal50; source-only weak2 promotion fixed before target; no test ranking',
  separate_head_training=False,test_views=['clean'],checkpoint_selection='final_only')
 return c

def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row,c.get('runtime_base',''),c.get('runtime_profile',''),c.get('runtime_metadata','')):raise ValueError('Original50/no-satellite feature treatment config differs')
 return c

def make_args(c,device='cuda:0'):
 validate(c);profile=read(c['runtime_profile']);a=baseline.make_args(c['model_seed'],device)
 a.wisig_pkl=profile['dataset']['manysig_pkl'];a.output_dir=c['output_root'];a.run_id=RUN;a.candidate_id=c['row_id'];a.num_workers=0
 baseline.assert_recipe(a)
 if not a.a1_source_screen_only or a.eval_sat_channel or a.sat_training_mode!='disabled':raise ValueError('Source-only training and disabled satellite paths required')
 return a

def expansion(source):
 if set(source)!=set(ARMS):raise ValueError('Complete initial4 source trajectories required')
 weak=sorted(sorted(range(6),key=lambda i:read(CALIBRATION)['risk'][i])[-2:]);base=source['BASE'];detail={};passed=[]
 for arm in ARMS[1:]:
  cur=source[arm];delta=[x-y for x,y in zip(cur['recall'],base['recall'])];gain=sum(delta[i] for i in weak)/2;overall=cur['accuracy']-base['accuracy'];worst=cur['worst_RX']-base['worst_RX'];ok=gain>=.002 and overall>=-.002 and worst>=-.005 and min(delta)>=-.01
  detail[arm]=dict(PASS=ok,weak_TX_gain_pp=100*gain,overall_pp=100*overall,worst_RX_pp=100*worst,per_TX_pp=[100*x for x in delta])
  if ok:passed.append(arm)
 return dict(status='SOURCE_EXPANSION_FIXED',weak_indices=weak,qualified=passed,expand_arms=['BASE',*passed] if passed else [],detail=detail,criterion='weak2 >=+0.2pp; overall>=-0.2pp; worstRX>=-0.5pp; anyTX>=-1pp vs local original50 BASE',target_access=False)
