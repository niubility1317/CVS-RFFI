"""One allowed-U ordering factor; original feature architecture, gates, EMA and default head."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_feature_multiscale50 import design as old
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,SEEDS,FULL_FP32_POLICY,read,write=(getattr(old,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','SEEDS','FULL_FP32_POLICY','read','write'))
RUN='20261014-phase1-neighbor-pair-oldgate-cos-e50-clean-manysig-m9-r02'
RELEASE='cvs_neighbor_pair50_20261014_r02';OWNER='codex/root/neighbor-pair50-20261014'
BASE=Path(PROJECT)/'runs'/RUN
ARMS=('OLD_GATE_COS50','RANDOM_ORDER','PAIRED_ORDER');BASELINE_ARM='OLD_GATE_COS50';PRIMARY='PAIRED_ORDER'
PAIRS=[('RANDOM_ORDER',BASELINE_ARM),('PAIRED_ORDER',BASELINE_ARM),('PAIRED_ORDER','RANDOM_ORDER')]
READOUTS=('ORIGINAL_HEAD',);VIEWS=('clean',);QUERY_COUNT=168000
SOURCE_DOMAINS=15;EPOCHS=50;LABEL_EPOCHS=33;PSEUDO_EPOCHS=17;STEPS=222;UPDATES=11100
BASELINE_RUN=old.RUN;BASELINE_EVAL_RUN=old.RUN
def rows():return [dict(row_id=f'{a}-s{s}',arm=a,model_seed=s) for s in SEEDS for a in ARMS]
def training_rows():return [r for r in rows() if r['arm']!=BASELINE_ARM]
def baseline_config(row):return old.config(dict(row_id=f"{BASELINE_ARM}-s{row['model_seed']}",arm=BASELINE_ARM,model_seed=row['model_seed']))
def config(row):
 if row not in rows():raise ValueError('Unregistered neighbor-pair row')
 prior=baseline_config(row);c=deepcopy(prior);arm=row['arm']
 c.update(schema='neighbor_pair_default_v1',run_id=RUN,**row,method='cvs_neighbor_pair50',stage='evaluation_reuse' if arm==BASELINE_ARM else 'neighbor_pair',output_root=(BASE/row['row_id']/'source').as_posix(),launch_owner=OWNER,source_config=prior if arm==BASELINE_ARM else None,model_initialization='historical comparison only' if arm==BASELINE_ARM else 'scratch',
  source_neighbor_sampling=dict(enabled=arm!=BASELINE_ARM,mode='PAIRS' if arm=='PAIRED_ORDER' else 'RANDOM',pool='original hidden U_s56700 only',allowed_fields=['base_index','day_i','eq_i','rx_i','sig_i'],batch=256,drop_last=False,all_packets_once=True,original_gates=True,extra_parameters=0,seed_policy='same one global Torch seed draw per native sampler iterator, private order generator'),
  readouts=list(READOUTS),classifier_training='default jointly trained only; no separate head fit',
  source_falsification=dict(first_seed=SEEDS[0],maximum_final_L_CE=.5,minimum_paired_U_acceptance=.25,metric='fixedfinal source LCE<=.5 both new controls and paired accepted U / total U>=.25 over fixed17pseudoepochs; no U truth, V/target ranking'),
  goal=dict(primary=PRIMARY,readout='ORIGINAL_HEAD',clean_three_seed_mean=.84,feedback_to_training=False),
  hypothesis_basis='Complete frozen-source neighbor-pair diagnostic: all3 source criteria pass, RANDOM reproduced360counts, V old-gate coverage3.410to89.962% with99.9959% precision; no target design feedback',
  source_qualification_run='20261014-phase1-source-neighbor-pair-gate-manysig-m3-r01',
  no_TIME_or_Sinc_variant_inheritance=True,classifier='unchanged ResidualPhysicalHead cosine6x160scale30 normal joint training',
  sampling_scope='Uniform random disjoint physical-neighbor pairs vs original uniform packet order; only U loader ordering, all gates/loss weights/model/EMA/input identical')
 c.pop('feature_extractor',None)
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row) or c['epochs']>50 or c['readouts']!=['ORIGINAL_HEAD']:raise ValueError('Fixed neighbor-pair/default-head matrix differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c)
 if c['arm']==BASELINE_ARM:raise ValueError('Historical baseline cannot be retrained')
 a=old.make_args(baseline_config(c),device);a.output_dir=c['output_root'];a.run_id=RUN;a.candidate_id=c['row_id'];a.seed=c['model_seed']
 return a
