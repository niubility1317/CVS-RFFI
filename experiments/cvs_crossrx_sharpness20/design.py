"""One scratch20 default-head matrix; first-seed source screening only."""
from pathlib import Path
from experiments.cvs_crossrx_credit50 import design as parent

ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,SEEDS,FULL_FP32_POLICY,read,write=(getattr(parent,k) for k in
 ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','SEEDS','FULL_FP32_POLICY','read','write'))
RUN='20261015-phase1-SIGTIME-center-crossRX-sharpness-e20-clean-manysig-m15-r01'
RELEASE='cvs_SIGTIME_center_crossRX_sharpness20_20261015_r01'
OWNER='codex/root/SIGTIME-center-crossRX-sharpness20-20261015'
BASE=Path(PROJECT)/'runs'/RUN
ARMS=('BASE_CE','BASE_SSL','CLEAN_CE','GLOBAL_GAP','CROSS_GAP')
PRIMARY=('CROSS_GAP',);VIEWS=('clean',);READOUTS=('NATIVE_HEAD',)
EPOCHS=20;STEPS=222;UPDATES=4440;WARMUP_STEPS=444;FREQUENCY=16;SCHEDULED_CALLS=250
PROBE_STEPS=(448,1776,3104,4432)
TITLE='E20跨RX特征脆弱性：SIG_TIME+固定CENTER25K默认头匹配矩阵'
PAIRS=[('CROSS_GAP','BASE_SSL'),('CROSS_GAP','GLOBAL_GAP'),('CROSS_GAP','CLEAN_CE'),('BASE_SSL','BASE_CE')]
SOURCE_GATE=dict(final_L_CE_max=.5,V_mean_drop_max=.01,V_worst_RX_drop_max=.02,
 eligible_fraction_min=.95,minimum_active_gradient_probes=2,
 minimum_positive_gap_call_fraction=.01,
 selection='All firstseed final20 source health; GLOBAL/CROSS actual auxiliary availability, no target ranking')

def rows():return [dict(row_id=f'{arm}-s{seed}',arm=arm,model_seed=seed) for seed in SEEDS for arm in ARMS]
def baseline_config(seed):return parent.baseline_config(seed)
def structural_config(seed):return parent.structural_config(seed)
def active(step):return WARMUP_STEPS<step<=UPDATES and step%FREQUENCY==0
def objective_mode(arm):return 'ZERO' if arm in ('BASE_CE','BASE_SSL') else arm

def config(row):
 if row not in rows():raise ValueError('Unregistered E20 sharpness row')
 base=structural_config(row['model_seed']);ssl=row['arm']!='BASE_CE'
 return dict(base,schema='phase1_crossRX_feature_sharpness20_v1',run_id=RUN,**row,
  launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),baseline_reference=None,baseline_reuse=False,
  epochs=EPOCHS,label_epochs=13 if ssl else 20,pseudo_epochs=7 if ssl else 0,updates=UPDATES,
  training_recipe='scratch20/cleanL/noordinaryIQaug; '+('13L+7U originalbatch_neighbor/ownEMA/entropy/cos' if ssl else '20LpureCE/UandEMAactuallydisabled/cos'),
  feature_extractor=dict(base['feature_extractor'],arm='SIG_TIME',sinc_coordinates='CENTER_25K',sinc_scale_hz=25000.,
   extra_mechanism_parameters=0,additional_classifier_parameters=0),
  training_mechanism=dict(mode=objective_mode(row['arm']),rho=.05,coefficient=.1,normalize_eps=1e-12,
   selected='primary identity sinc/time_fuse/t1/t2 + SIG_TIME sequence_pool excluding gain_raw',selected_tensors=17,
   warmup_steps=WARMUP_STEPS,frequency=FREQUENCY,scheduled_calls=SCHEDULED_CALLS,probe_steps=list(PROBE_STEPS),
   bank_role='L_s',bank_ordinals=[68,69],bank_count=180,current_global_base_index_exclusion=True,
   previously_trained_source_L=True,never_trained_holdout=False,unseen_RX_evaluation=False,
   loss='coefficient/5 sumRX relu(meanCEpert_otherRX-meanCEclean_otherRX); clean and pert feature graph, delta detached',
   default_W_gain_other_parameters_detached_in_auxiliary=True,main_default_head_joint_training=True,
   extra_actual_optimizer_steps=0,head_fit_updates=0,original_sampler_unchanged=True,
   all_controls_same_extra_forward_budget=True,missing_RX_policy='Skip entire scheduled auxiliary call and record; retain sampler/five denominator'),
  execution_overrides=dict(use_aug=False,use_concat_sat_channel_aug=False,use_sat_consistency=False,
   use_unlabeled=ssl,use_ema_teacher=ssl,label_smoothing=0,source_only=True),
  source_falsification=dict(SOURCE_GATE),test_views=['clean'],readouts=['NATIVE_HEAD'],separate_head_training=False,
  goal=dict(primary='CROSS_GAP',clean_three_seed_mean=.84,feedback_to_training=False),
  checkpoint_sources=[],model_initialization='scratch',checkpoint_selection='final_only',
  budget_authorization='Current AGENTS scratch20; fixed firstseed5/source screen then10. No oldE50 weight continuation.',
  baseline_definition='Matched SIG_TIME+CENTER scratch20 BASE_CE plus independent BASE_SSL. Original userE20 CE family retained as architecture-different historical reference.',
  cost_claim='15scratch20/300modelepoch/66600realupdates; firstseed5=100epoch/22200updates.250scheduled auxiliary calls/model, cleanbank+5temporaryfeature forwards pereligiblecall; actualextraL exposure counted.0newparams/heads/innerrealsteps.')

def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed E20 sharpness configuration differs')
 return c

def make_args(c,device='cuda:0'):
 validate(c);a=parent.make_args(parent.config(next(r for r in parent.rows() if r['arm']=='BASE' and r['model_seed']==c['model_seed'])),device)
 a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];a.epochs=EPOCHS
 a.label_epochs=c['label_epochs'];a.pseudo_epochs=c['pseudo_epochs'];a.use_aug=False
 a.use_unlabeled=c['execution_overrides']['use_unlabeled'];a.use_ema_teacher=c['execution_overrides']['use_ema_teacher']
 if not a.use_unlabeled:a.lambda_u=0.;a.lambda_ent=0.
 a.checkpoint_selection='final_only';return a

def source_reference_metrics(seed):return read(BASE/f'BASE_SSL-s{seed}'/'source/completion.json')['final_source_metrics']
def result_metadata():return dict(rows=15,new_backbones=15,existing_backbone_references=0,readouts=15,
 prediction_count=2520000,reused_native_decisions=0,new_prediction_decisions=2520000,metric_records=270,
 backbone_updates_new=66600,historical_baseline_updates=0)
