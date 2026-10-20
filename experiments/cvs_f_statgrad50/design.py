"""Original F50; two fixed statistic-gradient policies x three model seeds."""
from copy import deepcopy
from pathlib import Path
import json
from . import baseline
from experiments.cvs_f_supcon50 import design as original
ROOT=Path(__file__).resolve().parents[2]
RUN='20261020-phase1-FcrossRX-statGradient-e50-clean-local-m6-r01'
OWNER='codex/root/FcrossRX50-statGradient-local-20261020'
SEEDS=baseline.SEEDS;ARMS=('STAT_GRAD','NORM_ONLY');CLASSES=original.CLASSES
EPOCHS=50;LABEL_EPOCHS=33;PSEUDO_EPOCHS=17;STEPS=222;UPDATES=11100;VIEWS=('clean',)
FULL_FP32_POLICY=original.FULL_FP32_POLICY
AUTHORIZATION='Active weakTX feature-learning goal; user requires three model seeds, default head jointly trained, no satellite augmentation and clean-only testing'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def rows():return [dict(row_id=f'{a}-s{s}',treatment=a,model_seed=s) for s in SEEDS for a in ARMS]
def config(row,base,profile,metadata):
 if row not in rows() or Path(base).name!=RUN:raise ValueError('Fixed six scratch50 clean rows and fresh output only')
 c=deepcopy(baseline.original_config(row['model_seed']))
 c.update(schema='F_crossRX50_statistic_gradient_v1',run_id=RUN,row_id=row['row_id'],treatment=row['treatment'],launch_owner=OWNER,output_root=str(Path(base)/row['row_id']/'source'),runtime_base=str(base),runtime_profile=str(profile),runtime_metadata=str(metadata),baseline_reuse=False,
  statistic_gradient_policy=row['treatment'],contrastive_mode='CROSS_RX',contrastive_layers=['time_down','t2.pool'],contrastive_temperature=.2,contrastive_max_weight=.1,contrastive_warmup=5,contrastive_ramp=10,
  statistic_forward='Exact native current batch centered/std descriptors,eps1e-4; packetL2; same masks/logpositivecount/loss/weight; no running states',
  statistic_backward='STAT_GRAD: true current mean/std derivative; NORM_ONLY: native detached-stat direction times norm(full descriptor derivative)/norm(native descriptor derivative), ratio detached and no parameter-norm equivalence claim',
  weak_indices=[1,3],class_weights=[1.]*6,joint160_geometry=False,extra_model_forwards=0,extra_parameters=0,reference_local_run='20261019-phase1-FcrossRX-TXrisk-geometry-e50-clean-local-m4-r01',
  baseline_reference=dict(run_id=baseline.REFERENCE_RUN,commit=baseline.REFERENCE_COMMIT,arm='CROSS_RX',role='Historical method anchor only; actual comparison uses matched local BASE per seed; no weight inheritance'),
  training_recipe='Original F CROSS_RX50=33L+17U oldgate-ownEMA-cos RDrop1 LS.01 ordinaryAug;NO satellite;defaultheadjoint. Only shallow/middle SC statistic backward changes.',
  source_selection='Fixedfinal50; all six rows fixed before new target access, no source expansion gate',separate_head_training=False,test_views=['clean'],checkpoint_selection='final_only')
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row,c.get('runtime_base',''),c.get('runtime_profile',''),c.get('runtime_metadata','')):raise ValueError('Fixed statistic-gradient recipe differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c);profile=read(c['runtime_profile']);a=baseline.make_args(c['model_seed'],device);a.wisig_pkl=profile['dataset']['manysig_pkl'];a.output_dir=c['output_root'];a.run_id=RUN;a.candidate_id=c['row_id'];a.num_workers=0;baseline.assert_recipe(a)
 if not a.a1_source_screen_only or a.eval_sat_channel or a.sat_training_mode!='disabled':raise ValueError('Source-only training, no satellite')
 return a
def fixed_authorization():return dict(status='FIXED_THREE_SEED_MATRIX',user_authorization=AUTHORIZATION,source_gate_applied=False,fixed_rows=rows(),baseline_reused=True,target_feedback=False)
