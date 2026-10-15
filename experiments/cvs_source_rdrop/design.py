from pathlib import Path
from experiments.cvs_source_soft_relation import design as prior
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write=(getattr(prior,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write'))
parent=prior.parent;PARAMETERS=prior.PARAMETERS
RUN='20261015-phase1-source-dropout-consistency-manysig-s2026092701-r01'
RELEASE='cvs_source_rdrop_20261015_r01';OWNER='codex/root/source-rdrop-20261015';BASE=Path(PROJECT)/'runs'/RUN
QUALIFICATION=dict(source_only=True,KL_positive=True,KL_actual_t1_t2_SIG_nonzero_each_of5=True,
 global_gradient_cosine_with_paired_CE_min=0.,model_state_and_default_head_unchanged=True,
 no_automatic_training_release=True,on_pass='Only separately registered unchanged02/03 source check',
 on_fail='Preserve negative evidence; no coefficient, parent or dropout scans')
def rows():return [dict(row_id='SOURCE_RDROP-s2026092701',arm='SOURCE_RDROP',model_seed=2026092701)]
def source_config():return prior.source_config()
def config(row):
 if row not in rows():raise ValueError('Unregistered dropout source row')
 return dict(schema='source_rdrop_v1',run_id=RUN,**row,source_config=source_config(),parent_run=parent.RUN,
  parent_rule='Predefined lowest-seed SIG_TIME final50 control; not target-best',checkpoint_epoch=50,
  role='L_s',source_pool_count=6300,source_packets=900,cohorts=5,cohort_packets=180,
  source_cohorts='Fixed first five L180 cohorts,2 per6TXx5RXx3day;900unique physical samples',
  stochastic='Same strong IQ input .015 for two student train-mode forwards; all native dropout active; no BatchNorm',
  eval_reference='Same strong IQ with dropout inactive; no gradient or classifier fitting',
  objective='Mean two default CE with smoothing.01; auxiliary symmetric KL between two differentiable predictions',
  coefficient=1.,label_smoothing=.01,parameters=list(PARAMETERS),source_only=True,
  query_read=False,U_IQ_read=False,U_truth_read=False,V_forward=False,V_gradient=False,
  training_epochs=0,optimizer_steps=0,head_fit_updates=0,model_forward_packets=2700,
  strong_rng_seed=2026101519,qualification=QUALIFICATION.copy(),output_root=(BASE/row['row_id']).as_posix(),launch_owner=OWNER,
  basis='Existing source-only isolated dropout descriptor perturbation plus R-Drop general theory; actual two stochastic default predictions and identity-gradient compatibility not yet measured',
  limitation='Frozen final50 L source proxy; gradients and KL do not prove early training, U correctness, DG or84. Default classW derivative is measured but never fitted or updated.')
def validate(c):
 if c!=config(rows()[0]):raise ValueError('Fixed dropout source recipe differs')
expand_registration=prior.expand_registration
