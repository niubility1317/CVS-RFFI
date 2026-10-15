"""Two scratch E50 rows; no initialization from the failed semantic family."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_urban_branch50 import design as parent

ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,FORMAL_VIEWS,TRUTH,CLASSES=(getattr(parent,k) for k in
    ('PROJECT','SOURCE','FULL_FP32_POLICY','FORMAL_VIEWS','TRUTH','CLASSES'))
RUN='20261015-phase1-sigtime-bridge-teaching-e50-manysig-s2026092701-m2-r01'
RELEASE='cvs_sigtime_bridge_teaching_e50_20261015_r01'
OWNER='codex/root/bridge-teaching-e50-20261015'
BASE=Path(PROJECT)/'runs'/RUN
ARMS=('BRIDGE_CE','BRIDGE_SIG')
SCENES,VIEWS,URBAN,TRAIN_SCENES=parent.SCENES,parent.VIEWS,parent.URBAN,parent.TRAIN_SCENES
EPOCHS,STEPS,UPDATES=50,222,11100
read,write=parent.read,parent.write


def rows():return [dict(arm=a,model_seed=2026092701,row_id=f'{a}-s2026092701') for a in ARMS]


def references():
    from experiments.cvs_urban_semantic50 import design as semantic
    return dict(CLEAN=semantic.config(next(r for r in semantic.rows() if r['arm']=='D0')),
        NOMINAL=parent.config(parent.rows()[0]),
        POSITIVE_J=parent.config(next(r for r in parent.rows() if r['arm']=='A-J')))


def config(row):
    if row not in rows():raise ValueError('Unregistered two-row bridge pilot')
    c=deepcopy(parent.config(parent.rows()[0]))
    c.update(schema='sigtime_bridge_teaching_e50_v1',run_id=RUN,**row,method='cvs_urban_bridge50',launch_owner=OWNER,
        output_root=(BASE/row['row_id']/'source').as_posix(),baseline_reuse=False,reused=False,
        initialization='scratch',checkpoint_sources=[],parent_checkpoint=None,parent_training_epochs=0,
        ordinary_concat=True,curriculum=False,margin=False,urban_ssl=False,
        baseline_recipe='native generic+old gates+own EMA+cos; L33+U17; same L5580/dev720; original default head',
        bridge=dict(noise_improvement_db=6.,independent_noise=True,full_receiver_state=True,full_AGC=True,
            CE_coefficient=.25,base_CE_weights=[.5,.5],student_HQ_forward=True,teacher_clean_HQ_forward=True,
            original_recording_noise_removed=False,formal_IQ_changed=False,
            frozen='propagation/RX/day session/quality/lock/compensation/kernels/FIR delay/boundary/full AGC',
            train_stream=2026101401,calibration_stream=2026101508),
        auxiliary=dict(kind='clean_joint_anchor_plus_local_SIG' if row['arm']=='BRIDGE_SIG' else 'clean_joint_anchor',
            local_SIG=row['arm']=='BRIDGE_SIG',start_epoch=6,warmup_epochs=5,
            joint='half mean of clean-ownEMA to urban and clean-ownEMA to HQ direction losses',
            joint_reliability='own clean EMA correct on true L labels and urban scene only',
            SIG='full urban raw112 to independent same-H HQ ownEMA raw112; groups 6/15/6 per window; constant dt excluded',
            SIG_weight='full-view sigmoid((added-noise SNR-5)/5) times clean-ownEMA L correctness; no U truth',
            local_bound='coordinate smooth-L1/(1+smooth-L1); mean within each group then equal three-group mean',
            scales='own epoch5 HQ EMA per-group RMS on source dev720, floor1e-3; frozen thereafter',
            coefficient='own epoch5 once-only full feature norm ratio .1 to half satellite CE, bounded1e-4..10',
            calibration_matched_in_both_rows=True,head_aux_gradient=False,gain_aux_gradient=False,
            CE_updates_all=True,extra_U_objectives=[]),
        comparison_baseline='Two matched HQ rows; unchanged A0/A-J and matched clean-only final50 are read-only descriptive controls',
        source_expansion_rule='Final50 L_dev only: candidate clean vs matched D0>=.005; low vs A0>=.01; urban vs A0>=0; worst urban RX vs A0>=-.01. No target feedback. Local-S claim additionally needs SIG-CE low>=.01 with clean>=-.005.',
        expansion='Two-row single-model-seed pilot only; no semantic-family expansion or target-selected parents',
        parent_training_cost=0)
    return c


def validate(c):
    row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
    if row is None or c!=config(row):raise ValueError('Fixed same-H bridge pilot configuration differs')
    return c


def make_args(c,device='cuda:0'):
    validate(c);a=parent.make_args(parent.config(parent.rows()[0]),device)
    a.output_dir=c['output_root'];a.run_id=RUN;a.candidate_id=c['row_id']
    return a
