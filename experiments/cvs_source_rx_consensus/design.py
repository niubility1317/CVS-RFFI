"""Fixed source-only probe; no trained candidate or target feedback."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_source_risk_variance import design as first
from experiments.cvs_source_risk_variance_seeds import design as other
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write,expand_registration=(getattr(first,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write','expand_registration'))
RUN='20261015-phase1-source-rx-consensus-manysig-m3-r01'
RELEASE='cvs_source_rx_consensus_20261015_r01';OWNER='codex/root/source-rx-consensus-20261015';BASE=Path(PROJECT)/'runs'/RUN
SEEDS=(2026092701,2026092702,2026092703);RX=(1,3,4,6,8)
CLASS_W='id_backbone.encoder.core.id_backbone.cls_head.weight'
BODY=tuple(first.PARAMETERS[:3]);FEATURE_COUNT=195041;TAU=.5
QUALIFICATION=dict(source_only=True,models_required_for_future_training=3,minimum_global_masked_CE_cosine_each_RX=0.,global_min_RX_cosine_better_than_matched_random=True,global_retained_gradient_norm_ratio_min=.1,eligible_keep_fraction_each_cohort_min=.01,eligible_keep_fraction_each_cohort_max=.99,body_masked_gradient_nonzero_each_of5=True,model_and_default_W_unchanged=True,head_fit_updates=0,first_failure_skips_remaining_models=True,no_automatic_training_release=True)
def rows():return [dict(row_id=f'RX_CONSENSUS-s{s}',arm='RX_CONSENSUS',model_seed=s) for s in SEEDS]
def parent_design(row):return first.parent if row['model_seed']==SEEDS[0] else other.parent
def source_config(row):
 if row not in rows():raise ValueError('Unregistered consensus source row')
 return deepcopy(first.source_config() if row['model_seed']==SEEDS[0] else other.source_config(next(r for r in other.rows() if r['model_seed']==row['model_seed'])))
def config(row):
 if row not in rows():raise ValueError('Unregistered consensus source row')
 return dict(schema='source_rx_consensus_v1',run_id=RUN,**row,source_config=source_config(row),parent_run=parent_design(row).RUN,parent_rule='Own predefined original SIG_TIME control final50, never target-best',role='L_s',source_pool_count=6300,source_packets=900,cohorts=5,cohort_packets=180,source_cohorts='Original fixed firstfive L180 cohorts;2per6TXx5RXx3day,900unique physical IDs',receiver_groups=list(RX),label_smoothing=.01,strong_rng_seed=2026101537,random_mask_rng_seed=2026101538,stochastic='One original strong IQ .015/student train-mode/dropout forward per packet; no BatchNorm',feature_scope='Entire actual identity subtree, exclude class W alone; retain feature norms/gains/reference',feature_parameters=FEATURE_COUNT,body_parameters=list(BODY),class_W=CLASS_W,agreement_threshold=TAU,mask='abs(mean(sign(perRX CE gradient)))>=.5; five RX means at least4 agree; zeros abstain; no rescaling',matched_random='Same active count per actual parameter tensor, sampled among nonzero-eligible coordinates with private CPU RNG',proposed_training_boundary='L_s f_theta update only; ordinary default W and existing U gradients unchanged; NOT implemented or trained by this source probe',model_forward_packets=900,gradient_calls=30,optimizer_steps=0,head_fit_updates=0,U_IQ_read=False,U_truth_read=False,V_forward=False,V_gradient=False,query_read=False,qualification=deepcopy(QUALIFICATION),conditional_execution='Measure01 first; if sourceFAIL skip02/03; ifPASS measureboth02/03 unchanged regardlesstheirresults',output_root=(BASE/row['row_id']).as_posix(),launch_owner=OWNER,reference='https://arxiv.org/abs/2009.00329',limitation='Late frozen L source gradient diagnostic only; no update, U quality, DG, newTX or84 claim')
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed consensus source configuration differs')
