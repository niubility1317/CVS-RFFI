"""One-step source attribution, not a trained leave-receiver-out model."""
from pathlib import Path
from experiments.cvs_evidence_responsibility_train20 import design as parent

ROOT=Path(__file__).resolve().parents[2]
PROJECT,read,write=parent.PROJECT,parent.read,parent.write
RUN='20261015-phase1-SIGTIME-RX-blocked-norm-attribution-sourceL-s2026092701-r01'
RELEASE='cvs_SIGTIME_RX_blocked_norm_attribution_sourceL_20261015_r01'
SOURCE_RELEASE=PROJECT+'/releases/'+parent.RELEASE
OUTPUT=PROJECT+'/runs/'+RUN+'/source_diagnostic'
RX=(1,3,4,6,8);ORDINALS=(2,19,37,54)
MODES=('ZERO','JOINT_GAP','JOINT_NORM','ROUTED_GAP')
LR=2e-4

def request():
    return dict(run_id=RUN,output_root=OUTPUT,source_release=SOURCE_RELEASE,
        source_run=parent.RUN,source_row='BASE_CE-s2026092701',model_seed=2026092701,
        checkpoint='own initial_smoke.pt only; no trained final weights or optimizer moments',
        receivers=list(RX),current_ordinals=list(ORDINALS),held_ordinals=[66,67],modes=list(MODES),
        source_role='L_s',training_packets_per_cell=144,held_packets_per_cell=36,
        cells=80,temporary_AdamW_steps=160,network_IQ_forwards=5942,
        source_IQ_reads=900,actual_optimizer_updates=0,head_fit_updates=0,
        original_default_head_joint=True,LR=LR,weight_decay=1e-4,max_grad_norm=5.,
        auxiliary_weight=.1,auxiliary_tolerance=1e-6,loss_denominator=576,
        norm_control='Same numeric loss; detached factor norm(ROUTED_F)/norm(JOINT_F) applied only to JOINT backward',
        pairing='Each current bank excludes the held RX; norm factor and all training gradients use only its four RX',
        interpretation='Cold snapshot one-step RX-blocked attribution; not full E20 leave-RX DG and no main-run qualification',
        analysis='All20 cohorts and all5 held RX; report per-cell and per-RX ACTUAL/PRE_W identity CE, errors, F-gradient and actual-update norms',
        U_IQ_reads=0,V_IQ_reads=0,U_truth_read=False,target_access=False,
        analysis_code_commit='release_commit.txt')

def validate(value):
    expected=request();expected['analysis_code_commit']=value.get('analysis_code_commit')
    if value!=expected or not isinstance(value.get('analysis_code_commit'),str) or len(value['analysis_code_commit'])!=40:
        raise ValueError('Fixed RX-blocked attribution request differs')
    return value
