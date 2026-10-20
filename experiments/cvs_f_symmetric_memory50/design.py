"""Fixed F50 crossRX livecurrent-key short-history memory study, clean only."""
from copy import deepcopy
from pathlib import Path
import json
from . import baseline
from experiments.cvs_f_supcon50 import design as original
ROOT=Path(__file__).resolve().parents[2]
RUN='20261020-phase1-FcrossRX-symmetricMemory-e50-clean-local-m1-r01'
OWNER='codex/root/FcrossRX50-symmetricMemory-local-20261020'
SEEDS=baseline.SEEDS;ARMS=('SYM4',);CLASSES=original.CLASSES
EPOCHS=50;LABEL_EPOCHS=33;PSEUDO_EPOCHS=17;STEPS=222;UPDATES=11100;VIEWS=('clean',)
FULL_FP32_POLICY=original.FULL_FP32_POLICY
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def rows(seeds=SEEDS):return [dict(row_id=f'{a}-s{s}',treatment=a,model_seed=s) for s in seeds for a in ARMS]
def config(row,base,profile,metadata):
 if row not in rows() or Path(base).name!=RUN:raise ValueError('Registered original F50 clean rows/output only')
 c=deepcopy(baseline.original_config(row['model_seed']))
 c.update(schema='F_crossRX50_crossbatch_memory_local_v1',run_id=RUN,row_id=row['row_id'],treatment=row['treatment'],launch_owner=OWNER,output_root=str(Path(base)/row['row_id']/'source'),runtime_base=str(base),runtime_profile=str(profile),runtime_metadata=str(metadata),baseline_reuse=False,
  weak_indices=[1,3],contrastive_mode=row['treatment'],contrastive_layers=['time_down','t2.pool'],contrastive_temperature=.2,contrastive_max_weight=.1,contrastive_warmup=5,contrastive_ramp=10,contrastive_positive='same TX different RX, distinct sourceL physical packets in current or last3 L batches',contrastive_negative='different TX, all source RX',current_keys_detached=False,history_keys_detached=True,memory_previous_steps=3,memory_deduplicate_physical_ID=True,memory_normalization='current detached batch center/std for anchors and raw current/history keys',contrastive_class_reweight=False,
  class_weights=[1.]*6,joint160_geometry=False,joint160_max_weight=0.,reference_local_run='20261019-phase1-FcrossRX-TXrisk-geometry-e50-clean-local-m4-r01',
  baseline_reference=dict(run_id=baseline.REFERENCE_RUN,commit=baseline.REFERENCE_COMMIT,arm='CROSS_RX',clean_mean_percent=85.11388888888888,role='Historical method anchor; reuse exact matching fresh local BASE; no model inheritance'),
  training_recipe='Original F CROSS_RX50=33L+17U oldgate-ownEMA-cos RDrop1 LS.01 ordinaryAug;NO satellite;defaultheadjoint; substitute shallow/middle SC with livecurrent/detachedpast key SYM4',
  source_falsification=None,source_selection='Fixedfinal50; source-only default-head each-weak improvement and otherTX health before target',separate_head_training=False,test_views=['clean'],checkpoint_selection='final_only')
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row,c.get('runtime_base',''),c.get('runtime_profile',''),c.get('runtime_metadata','')):raise ValueError('Fixed crossbatch memory recipe differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c);profile=read(c['runtime_profile']);a=baseline.make_args(c['model_seed'],device);a.wisig_pkl=profile['dataset']['manysig_pkl'];a.output_dir=c['output_root'];a.run_id=RUN;a.candidate_id=c['row_id'];a.num_workers=0;baseline.assert_recipe(a)
 if not a.a1_source_screen_only or a.eval_sat_channel or a.sat_training_mode!='disabled':raise ValueError('Source-only train, no satellite')
 return a
def expansion(current,base):
 decisions={}
 for arm,value in current.items():
  delta=[a-b for a,b in zip(value['recall'],base['recall'])];weak=all(delta[c]>0 for c in (1,3));others=all(delta[c]>=-.0025 for c in (0,2,4,5));health=value['accuracy']>=base['accuracy'] and value['worst_RX']>=base['worst_RX']-.0025 and all(v['zero_feature_count']==0 for v in value['geometry']['per_TX'])
  decisions[arm]=dict(PASS=weak and others and health,each_weak_improved=weak,otherTX_health=others,source_classification_health=health,overall_pp=100*(value['accuracy']-base['accuracy']),worst_RX_pp=100*(value['worst_RX']-base['worst_RX']),per_TX_pp=[100*x for x in delta])
 return dict(status='SOURCE_EXPANSION_FIXED',arms=decisions,expand_arms=[a for a in ARMS if a in decisions and decisions[a]['PASS']],target_access=False,criterion='Full sourceV27000 defaulthead fixedfinal50: BOTH weakTX recall strictly improve over same-seed BASE; eachotherTX >=BASE-.25pp, overall>=BASE, worstRX>=BASE-.25pp, no zero features. All initial rows still clean-tested, expansion fixed before any new target results.')
