"""Source-fixed weak-TX balanced tail feature risk; original50 F+CROSS_RX controls reused."""
from copy import deepcopy
from pathlib import Path
import json
from . import baseline
from experiments.cvs_f_supcon50 import design as original
ROOT=Path(__file__).resolve().parents[2]
RUN='20261020-phase1-FcrossRX-weakTX-tailRisk-e50-clean-local-m2-r01'
OWNER='codex/root/FcrossRX50-weakTX-tailRisk-local-20261020'
SEEDS=baseline.SEEDS;ARMS=('MEAN_WEAK_RX','TAIL_WEAK_RX');CLASSES=original.CLASSES
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
 if cal['class_names']!=list(CLASSES) or cal['calibration_role']!='sourceV only' or cal['target_feedback']:raise ValueError('Prior source-only weak TX calibration required')
 c.update(schema='F_crossRX50_weakTX_tail_risk_local_v1',run_id=RUN,row_id=row['row_id'],treatment=row['treatment'],launch_owner=OWNER,output_root=str(Path(base)/row['row_id']/'source'),runtime_base=str(base),runtime_profile=str(profile),runtime_metadata=str(metadata),baseline_reuse=False,
  weak_indices=[1,3],tail_fraction=.5,tail_max_weight=.1,tail_warmup=5,tail_ramp=10,tail_levels=['time_down','t2.pool'],tail_RX_balanced=True,tail_pair_average_before_selection=True,tail_temperature=.2,tail_negative_RX='sameRX',
  class_weights=[1.]*6,joint160_geometry=False,joint160_max_weight=0.,reference_local_run='20261019-phase1-FcrossRX-TXrisk-geometry-e50-clean-local-m4-r01',
  baseline_reference=dict(run_id=baseline.REFERENCE_RUN,commit=baseline.REFERENCE_COMMIT,arm='CROSS_RX',clean_mean_percent=85.11388888888888,role='Historical method anchor. Reuse matching fresh local BASE sources/predictions; no inherited model state'),
  training_recipe='Original F CROSS_RX50=33L+17U oldgate-ownEMA-cos RDrop1 LS.01 ordinaryAug;NO satellite;defaultheadjoint; current-L paired shallow/middle weakTX/RX mean or upper-half feature risk',
  source_falsification=None,source_selection='Fixedfinal50; source-only each-weak feature geometry and default-head health promotion before target; no test ranking',separate_head_training=False,test_views=['clean'],checkpoint_selection='final_only')
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row,c.get('runtime_base',''),c.get('runtime_profile',''),c.get('runtime_metadata','')):raise ValueError('Fixed source-only weak feature tail recipe differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c);profile=read(c['runtime_profile']);a=baseline.make_args(c['model_seed'],device);a.wisig_pkl=profile['dataset']['manysig_pkl'];a.output_dir=c['output_root'];a.run_id=RUN;a.candidate_id=c['row_id'];a.num_workers=0;baseline.assert_recipe(a)
 if not a.a1_source_screen_only or a.eval_sat_channel or a.sat_training_mode!='disabled':raise ValueError('Source-only train, no satellite')
 return a

BASELINE_GEOMETRY=ROOT/'automation_reports/CV-SincNet/20261019-diagnostic-FcrossRX-feature-geometry-sourceV-m3-local-r01/result.json'
def reference_geometry(seed):
 r=read(BASELINE_GEOMETRY)
 if r['reference_run']!='20261019-phase1-FcrossRX-TXrisk-geometry-e50-clean-local-m4-r01' or r['reference_code_commit']!='31467086c3ced30237f3ac86e91eae68645425d7' or r['target_reads'] or r['head_fit_updates'] or r['prototype_classifier_decisions']:raise ValueError('Prior legal sourceV geometry only')
 g=next(v['geometry'] for v in r['records'] if v['model_seed']==seed)
 if g['class_names']!=list(CLASSES) or g['count']!=27000 or g['dimensions']!=160:raise ValueError('Matching actual joint160 source geometry required')
 return g

def expansion(current,base):
 old=reference_geometry(SEEDS[0]);decisions={}
 for arm,value in current.items():
  delta=[a-b for a,b in zip(value['recall'],base['recall'])];g=value['geometry'];details=[]
  for c in (1,3):
   now=g['per_TX'][c];ref=old['per_TX'][c];gain=now['out_RX_margin']['q10']-ref['out_RX_margin']['q10'];ratio=now['RX_spread_to_inter_TX_gap']/ref['RX_spread_to_inter_TX_gap'];details.append(dict(TX=CLASSES[c],out_RX_margin_q10_gain=gain,RX_spread_to_gap_ratio=ratio,PASS=gain>=.002 and ratio<=1.05 and now['zero_feature_count']==0))
  healthy=value['tail_scheduled']==9990 and value['tail_eligible_fraction']>=.99 and value['tail_cell_coverage']>=.95 and value['accuracy']-base['accuracy']>=-.002 and value['worst_RX']-base['worst_RX']>=-.005 and min(delta)>=-.01
  decisions[arm]=dict(PASS=healthy and all(v['PASS'] for v in details),source_classification_health=healthy,overall_pp=100*(value['accuracy']-base['accuracy']),worst_RX_pp=100*(value['worst_RX']-base['worst_RX']),per_TX_pp=[100*x for x in delta],weak_geometry=details)
 return dict(status='SOURCE_EXPANSION_FIXED',arms=decisions,expand_arms=[a for a,v in decisions.items() if v['PASS']],target_access=False,criterion='Each weakTX outRX feature-margin q10 >= same-seed BASE+.002; eligible feature calls>=99%; weakTX/RX cell coverage>=95%; RXspread/interclassgap <=1.05x BASE; no degenerate features; defaulthead overall>=-.2pp,worstRX>=-.5pp,anyTX>=-1pp. Source only, no retrofit of prior rejected candidates.')
