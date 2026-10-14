from pathlib import Path
from experiments.cvs_source_self_challenge import design as prior
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write=(getattr(prior,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write'))
parent=prior.parent
RUN='20261015-phase1-source-soft-relation-manysig-s2026092701-r01'
RELEASE='cvs_source_soft_relation_20261015_r01'
OWNER='codex/root/source-soft-relation-20261015'
BASE=Path(PROJECT)/'runs'/RUN
PARAMETERS=prior.PARAMETERS
QUALIFICATION=dict(source_only=True,teacher_soft_edge_true_identity_mass_min=.90,
    XR_actual_t1_t2_SIG_nonzero_each_cohort=True,XR_global_gradient_cosine_with_matched_strong_CE_min=0.,
    model_state_and_default_head_unchanged=True,no_automatic_training_release=True,
    on_pass='Only separately registered unchanged02/03 source diagnostic, not training',
    on_fail='Retain result; do not scan temperatures, weights or parent checkpoints')
def rows():return prior.rows()
def source_config():return prior.source_config()
def config(row):
 if row not in rows():raise ValueError('Unregistered source semantic relation row')
 return dict(schema='source_soft_relation_v1',run_id=RUN,**row,source_config=source_config(),
    parent_run=parent.RUN,parent_rule='Predefined SIG_TIME first nominalseed final50 control; never target-best',
    checkpoint_epoch=50,role='L_s',source_pool_count=6300,source_packets=900,cohorts=5,cohort_packets=180,
    source_cohorts='Unchanged first five L180 cohorts;2perTX/RX/day;900unique physical packets',
    student='original student train mode/strong IQ noise.015/dropout; no BatchNorm mutation',
    teacher='ownEMA0.999 stored in same final50 checkpoint; eval/weak/detached',
    modes=['ALL_GRAPH','XR_GRAPH'],temperature=.2,coefficient=.16,label_smoothing=.01,
    graph_target='Row-normalized detached teacher class-probability dot products; no self-edge; XR only across sourceRX',
    graph_prediction='Row-softmax deployed student normalizedjoint160 cosine / temperature on identical allowed edges',
    loss='mean row KL(target||feature-edge prediction), zero undefined rows; no truth/threshold/projection/head',
    parameters=list(PARAMETERS),source_only=True,query_read=False,U_IQ_read=False,U_truth_read=False,
    V_forward=False,V_gradient=False,training_epochs=0,optimizer_steps=0,head_fit_updates=0,
    model_forward_packets=1800,strong_rng_seed=2026101507,qualification=QUALIFICATION.copy(),
    output_root=(BASE/row['row_id']).as_posix(),launch_owner=OWNER,
    basis='Existing same-instance MT actual source gradients small/nearorthogonal; semantic pair graph is a distinct objective inspired by CoMatch, not a reproduction or target-driven RSC retuning',
    limitation='Frozen final50 L proxy only; cannot infer U truth, early training or DG/84; default head never separately fitted')
def validate(c):
 if c!=config(rows()[0]):raise ValueError('Fixed source soft-relation recipe differs')
def expand_registration(value):
 if isinstance(value,str):return PROJECT+value[len('PROJECT'):] if value=='PROJECT' or value.startswith('PROJECT/') else value
 if isinstance(value,list):return [expand_registration(v) for v in value]
 if isinstance(value,dict):return {k:expand_registration(v) for k,v in value.items()}
 return value
