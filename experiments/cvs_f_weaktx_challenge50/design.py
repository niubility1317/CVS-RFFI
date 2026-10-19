"""Source-fixed weak-TX feature challenge; original50 F+CROSS_RX controls reused."""
from copy import deepcopy
from pathlib import Path
import json
from . import baseline
from experiments.cvs_f_supcon50 import design as original
ROOT=Path(__file__).resolve().parents[2]
RUN='20261020-phase1-FcrossRX-weakTX-challenge-e50-clean-local-m2-r01'
OWNER='codex/root/FcrossRX50-weakTX-challenge-local-20261020'
SEEDS=baseline.SEEDS;ARMS=('RANDOM_WEAK','SALIENCY_WEAK');CLASSES=original.CLASSES
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
 c.update(schema='F_crossRX50_weakTX_challenge_local_v1',run_id=RUN,row_id=row['row_id'],treatment=row['treatment'],launch_owner=OWNER,output_root=str(Path(base)/row['row_id']/'source'),runtime_base=str(base),runtime_profile=str(profile),runtime_metadata=str(metadata),baseline_reuse=False,
  weak_indices=[1,3],challenge_levels=['time_down','t2.pool'],challenge_drop_fraction=.125,challenge_every=4,challenge_max_weight=.1,challenge_warmup=5,challenge_ramp=10,
  class_weights=[1.]*6,joint160_geometry=False,joint160_max_weight=0.,reference_local_run='20261019-phase1-FcrossRX-TXrisk-geometry-e50-clean-local-m4-r01',
  baseline_reference=dict(run_id=baseline.REFERENCE_RUN,commit=baseline.REFERENCE_COMMIT,arm='CROSS_RX',clean_mean_percent=85.11388888888888,role='Historical method anchor. Reuse matching fresh local BASE sources/predictions; no inherited model state'),
  training_recipe='Original F CROSS_RX50=33L+17U oldgate-ownEMA-cos RDrop1 LS.01 ordinaryAug;NO satellite;defaultheadjoint; source-fixed weak-TX shallow/mid channel challenge',
  source_falsification=None,source_selection='Fixedfinal50; source-only weak2 promotion fixed before target; no test ranking',separate_head_training=False,test_views=['clean'],checkpoint_selection='final_only')
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row,c.get('runtime_base',''),c.get('runtime_profile',''),c.get('runtime_metadata','')):raise ValueError('Fixed source-only feature challenge recipe differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c);profile=read(c['runtime_profile']);a=baseline.make_args(c['model_seed'],device);a.wisig_pkl=profile['dataset']['manysig_pkl'];a.output_dir=c['output_root'];a.run_id=RUN;a.candidate_id=c['row_id'];a.num_workers=0;baseline.assert_recipe(a)
 if not a.a1_source_screen_only or a.eval_sat_channel or a.sat_training_mode!='disabled':raise ValueError('Source-only train, no satellite')
 return a
def expansion(current,base):
 from experiments.cvs_f_rxmargin50.design import expansion as rule
 decisions={a:rule(current[a],base) for a in ARMS}
 return dict(status='SOURCE_EXPANSION_FIXED',arms=decisions,expand_arms=[a for a,v in decisions.items() if v['PASS']],target_access=False,criterion='Each arm: weak2>=+0.2pp; overall>=-0.2pp; worstRX>=-0.5pp; anyTX>=-1pp. Fixed before any new candidate target.')
