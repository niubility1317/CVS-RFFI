from pathlib import Path
from experiments.cvs_sigtime_rdrop_envelope20 import design as parent

ROOT = Path(__file__).resolve().parents[2]
PROJECT, SOURCE, read, write, FULL_FP32_POLICY = (getattr(parent, k) for k in ('PROJECT', 'SOURCE', 'read', 'write', 'FULL_FP32_POLICY'))
RUN = '20261016-phase1-SIGTIME-rdrop-operator-gradient-source-s2026092701-r01'
RELEASE = 'cvs_SIGTIME_rdrop_operator_probe_20261016_r01'
OWNER = 'codex/root/SIGTIME-RDrop-operator-probe-20261016'
BASE = Path(PROJECT) / 'runs' / RUN
PARENT_RELEASE = PROJECT + '/releases/' + parent.RELEASE
PARENT_COMMIT = '7827a9a6b04764e690ecaf90457666435017c824'
ROW = next(r for r in parent.rows() if r['arm'] == 'FULL')
CONTEXTS = ('CLEAN', 'NATIVE_AUG')
DRAW_SEEDS = (2026101601, 2026101602, 2026101603, 2026101604)
PARAMETERS = (
    'id_backbone.encoder.core.id_backbone.t1.pw.weight',
    'id_backbone.encoder.core.id_backbone.t2.pw.weight',
    'id_backbone.encoder.sequence_pool.local_projection.weight',
    'id_backbone.encoder.core.id_backbone.cls_head.weight',
)


def request():
    return dict(run_id=RUN, launch_owner=OWNER, source_only=True, target_access=False,
        parent_release=PARENT_RELEASE, parent_commit=PARENT_COMMIT, parent_row=ROW,
        model_release=parent.MODEL_RELEASE, dependency_release=parent.DEPENDENCY_RELEASE,
        fold_release=parent.FOLD_RELEASE, checkpoint=(Path(parent.config(ROW)['output_root'])/'final_ssdg.pth').as_posix(),
        output_root=BASE.as_posix(), contexts=list(CONTEXTS), draw_seeds=list(DRAW_SEEDS),
        source_selection='First2 native L indices in each TX/RX/day cell,fit RX3/4/6/8 only;144 unique L',
        input_views='Clean plus original native normal augmentation configured at min(final20,label13),each augmentation drawn once and reused in both forwards',
        dropout='Same paired native identity-only outputs reused by all4 operators;common seed across clean/aug contexts',
        parameters=list(PARAMETERS), operators=['FULL','MEAN','CONFIDENCE','ENVELOPE'],
        physical_L_materializations=144, synthetic_checkpoint_smoke_packets=2,
        actual_identity_forward_packets=2304, optimizer_steps=0, head_fit_updates=0,
        U_IQ_reads=0, V_IQ_reads=0, held_RX1_IQ_reads=0, target_IQ_reads=0,
        interpretation='Local gradient direction and magnitude on seen fit L and frozen model;not an AdamW update,causal attribution,validation performance or next-candidate selection threshold')


def validate(c):
    if c != request():
        raise ValueError('Only fixed source gradient diagnostic;no new checkpoint,IQ scope,operator or update')
    return c
