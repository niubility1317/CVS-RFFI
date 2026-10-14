"""One matched scratch SIG_TIME training-mechanism matrix, default head only."""
from pathlib import Path
from experiments.cvs_feature_sequence50 import design as parent

ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,SEEDS,FULL_FP32_POLICY,read,write=(getattr(parent,k) for k in
 ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','SEEDS','FULL_FP32_POLICY','read','write'))
RUN='20261014-phase1-SIGTIME-credit-oldgate-e50-clean-manysig-m12-r01'
RELEASE='cvs_SIGTIME_credit50_20261014_r01';OWNER='codex/root/SIGTIME-credit-training-20261014'
BASE=Path(PROJECT)/'runs'/RUN
ARMS=('BASE','SUPCON','ORDER','CREDIT');PRIMARY=('CREDIT',)
VIEWS=('clean',);READOUTS=('NATIVE_HEAD',)
TITLE='SIG_TIME中层停止排序与实际AdamW跨RX增量：默认头匹配消融'
PAIRS=[('CREDIT','BASE'),('ORDER','BASE'),('SUPCON','BASE'),('CREDIT','ORDER'),('CREDIT','SUPCON')]
EPOCHS=50;LABEL_EPOCHS=33;PSEUDO_EPOCHS=17;STEPS=222;UPDATES=11100
PROBE_STEPS=(1120,4448,7776,11088)
SOURCE_GATE=dict(final_L_CE_max=.5,V_mean_drop_max=.01,V_worst_RX_drop_max=.02,
 eligible_fraction_min=.95,CREDIT_accepted_group_fraction_min=.01,
 minimum_active_gradient_probes=2,
 selection='All firstseed source health and actual-mechanism availability; never target ranking.')

def rows():return [dict(row_id=f'{arm}-s{seed}',arm=arm,model_seed=seed) for seed in SEEDS for arm in ARMS]
def baseline_config(seed):return parent.baseline_config(seed)
def structural_config(seed):return parent.config(next(r for r in parent.rows() if r['arm']=='SIG_TIME' and r['model_seed']==seed))
def active(step):return 1110<step<=11100 and step%16==0

def config(row):
 if row not in rows():raise ValueError('Unregistered cross-RX training row')
 base=structural_config(row['model_seed'])
 return dict(base,schema='phase1_SIGTIME_crossRX_credit50_v1',run_id=RUN,**row,
  launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),baseline_reference=None,
  baseline_reuse=False,feature_extractor=dict(base['feature_extractor'],arm='SIG_TIME',extra_mechanism_parameters=0),
  training_mechanism=dict(mode=row['arm'],layer='sequence_pool.output_norm',layer_kind='SIG_TIME_pre_gain160',
   role='L_s',coefficient=.1,margin=.1,temperature=.1,normalize_eps=1e-4,credit_tolerance=1e-6,
   warmup_steps=1110,frequency=16,scheduled_calls=624,probe_steps=list(PROBE_STEPS),
   references='Detached current L sameTX/day crossRX positive and same referenceRX/day otherTX negative',
   denominator='FULL legal triplet count including satisfied terms; no gate-active reweighting',
   utility='Full assembled native base loss (including contemporaneous U); same actual preAdamW state, independentclip5, same preW default readout on other seenL RX',
   late_single_addition=True,default_head_joint_training=True,auxiliary_W_and_gain_None=True,
   original_L_U_sampler_unchanged=True,original_pseudo_gates_EMA_entropy_unchanged=True,
   all_controls_same_counterfactual_forward_budget=True,missing_RX_policy='Skip entire scheduled auxiliary call; record and retain original sampler',
   extra_actual_optimizer_steps=0,new_parameters=0,new_heads=0,head_fit_updates=0),
  source_falsification=dict(SOURCE_GATE),test_views=['clean'],readouts=['NATIVE_HEAD'],
  separate_head_training=False,goal=dict(primary='CREDIT',clean_three_seed_mean=.84,feedback_to_training=False),
  checkpoint_sources=[],model_initialization='scratch',
  budget_authorization='Current goal permits <=50 scratch per backbone and latest f5 report permits one main training matrix. Source final-state probe completed; this is early/mid actual training efficacy test with fixed source rules.',
  cost_claim='12scratch50/133200realupdates; allSIG_TIME initialarchitecture/ownEMA/sourceL-U data and original U schedule.624scheduledcalls/model, fiveRX two shadow steps/call plus four actual-shadow checks; extra sourceeval forwards disclosed, no newheads/params/realupdates.')

def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed SIG_TIME training mechanism/config differs')
 return c

def make_args(c,device='cuda:0'):
 validate(c);a=parent.make_args(structural_config(c['model_seed']),device)
 a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];return a

def source_reference_metrics(seed):
 return read(BASE/f'BASE-s{seed}'/'source/completion.json')['final_source_metrics']

def result_metadata():return dict(rows=12,new_backbones=12,existing_backbone_references=0,readouts=12,
 prediction_count=2016000,reused_native_decisions=0,new_prediction_decisions=2016000,
 metric_records=216,backbone_updates_new=133200,historical_baseline_updates=0)
