from pathlib import Path
from experiments.cvs_source_rdrop import design as old
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write=(getattr(old,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write'))
parent=old.parent;PARAMETERS=old.PARAMETERS;expand_registration=old.expand_registration
RUN='20261015-phase1-source-risk-variance-manysig-s2026092701-r01'
RELEASE='cvs_source_risk_variance_20261015_r01';OWNER='codex/root/source-risk-variance-20261015';BASE=Path(PROJECT)/'runs'/RUN
QUALIFICATION=dict(source_only=True,positive_risk_variance_each_of5=True,actual_t1_t2_SIG_nonzero_each_of5=True,global_gradient_cosine_with_balanced_CE_min=0.,model_state_and_default_head_unchanged=True,no_automatic_training_release=True,on_pass='Only separately registered unchanged02/03 source check',on_fail='Retain negative; no coefficient, grouping, parent or threshold scans')
def rows():return [dict(row_id='SOURCE_VREX-s2026092701',arm='SOURCE_VREX',model_seed=2026092701)]
def source_config():return old.source_config()
def config(row):
 if row not in rows():raise ValueError('Unregistered risk variance source row')
 return dict(schema='source_risk_variance_v1',run_id=RUN,**row,source_config=source_config(),parent_run=parent.RUN,parent_rule='Original predefined lowest seed SIG_TIME final50 control, never target-best',checkpoint_epoch=50,role='L_s',source_pool_count=6300,source_packets=900,cohorts=5,cohort_packets=180,source_cohorts='Existing fixed firstfive L180 cohorts,2per6TXx5RXx3day,900unique physical samples',receiver_groups=[1,3,4,6,8],stochastic='One original student train-mode strong IQ noise.015 forward per source packet; dropout active, no BatchNorm',objective='Mean five receiver CE risks vs population variance of those same five differentiable risks; smoothing.01',coefficient=1.,label_smoothing=.01,parameters=list(PARAMETERS),source_only=True,query_read=False,U_IQ_read=False,U_truth_read=False,V_forward=False,V_gradient=False,training_epochs=0,optimizer_steps=0,head_fit_updates=0,model_forward_packets=900,strong_rng_seed=2026101529,qualification=QUALIFICATION.copy(),output_root=(BASE/row['row_id']).as_posix(),launch_owner=OWNER,basis='GroupDRO risk weighting did not establish stable gain; V-REx instead penalizes variance of legal receiver risks. Unlike Fishr, no variance of parameter gradients or higher-order derivative objective. Source derivative compatibility not assumed.',limitation='Frozen late-stage L proxy; does not prove early training, U correctness, DG or84; existing W derivative measured but never fitted or updated')
def validate(c):
 if c!=config(rows()[0]):raise ValueError('Fixed source risk variance recipe differs')
