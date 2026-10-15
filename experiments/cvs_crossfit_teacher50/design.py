"""Two independent equal-budget teachers; no library/student expansion here."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_urban_branch50 import design as parent

ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY=(getattr(parent,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY'))
RUN='20261015-phase1-crossfit-RX1-teachers-e50-manysig-s2026092701-m2-r01'
RELEASE='cvs_crossfit_RX1_teachers_e50_20261015_r01'
OWNER='codex/root/crossfit-RX1-teachers-20261015'
BASE=Path(PROJECT)/'runs'/RUN
ARMS=('SAMPLE','EXCLUDE_RX1')
HELD_RX=1
MODEL_SEED=2026092701
SUBSET_SEED=2026101505
EVALUATION_SEED=2026101506
EPOCHS,STEPS,UPDATES=50,222,11100
read,write=parent.read,parent.write
SCENES,URBAN=parent.SCENES,parent.URBAN


def rows():return [dict(arm=a,model_seed=MODEL_SEED,row_id=f'{a}-s{MODEL_SEED}') for a in ARMS]


def config(row):
    if row not in rows():raise ValueError('Only fixed RX1 two-teacher pilot authorized here')
    c=deepcopy(parent.config(parent.rows()[0]))
    c.update(schema='sigtime_crossfit_teacher50_v1',run_id=RUN,**row,method='cvs_crossfit_teacher50',launch_owner=OWNER,
        output_root=(BASE/row['row_id']/'source').as_posix(),label_epochs=50,pseudo_epochs=0,
        baseline_recipe='independent scratch teacher: L_train subset clean CE only; native cos/head joint',
        ordinary_concat=False,paired=False,train_scenes=[],curriculum=False,margin=False,urban_ssl=False,
        source_roles=dict(L_s=6300,L_outer_train=5580,L_teacher_train=4464,L_dev=720,L_unused=1116,U_s=56700,V=27000),
        teacher_training=dict(held_RX=HELD_RX,subset_seed=SUBSET_SEED,train_count=4464,all_TX=True,
            input='clean native RMS IQ256',unlabeled=False,EMA=False,head_fit=False,auxiliary_losses=[],
            original_recording_TX_hardware_inverted=False,independent_teacher_feature_alignment=False),
        auxiliary=dict(position='none',head_aux_gradient=False,gain_aux_gradient=False),
        checkpoint_selection='final_only',test_views=[],
        evaluation=dict(source_only=True,source_role='same L_dev720',primary_RX=1,primary_count=144,
            seed=EVALUATION_SEED,views=['clean',*SCENES],target_access=False,
            baseline_gate='fixed A0 final50 student/ownEMA native batch-neighbor and strong-agreement replay; source replay not actual U decisions',
            primary='EXCLUDE_RX1 corrections minus destruction vs SAMPLE on RX1 clean; no source labels select per-sample teacher',
            expansion='No automatic bank/student: source RX1 clean net>=3/144, native-gate-rejected net>=0, highconfidence-wrong increase<=.02; otherwise retain negative result'),
        comparison_baseline='SAMPLE uses equal4464 labeled physical items and exact same TX/day counts, all5 RX; EXCLUDE uses four RX',
        expansion='Two teachers only; later five-pair bank and BASE/SAMPLE-KD/RX-KD student not launched',
        benchmark_context='Source-only teacher falsification on exposed benchmark; no new target result')
    return c


def validate(c):
    row=next((v for v in rows() if v['row_id']==c.get('row_id')),None)
    if row is None or c!=config(row):raise ValueError('Fixed source teacher recipe differs')
    return c


def make_args(c,device='cuda:0'):
    validate(c);a=parent.make_args(parent.config(parent.rows()[0]),device)
    a.output_dir=c['output_root'];a.run_id=RUN;a.candidate_id=c['row_id']
    a.epochs=50;a.label_epochs=50;a.pseudo_epochs=0;a.use_unlabeled=False;a.use_ema_teacher=False
    for name in vars(a):
        if name.startswith('lambda_'):setattr(a,name,0.)
    a.use_concat_sat_channel_aug=False;a.sat_training_mode='disabled';a.use_sat_consistency=False;a.use_aug=False
    a.checkpoint_selection='final_only';a.test_eval_start_epoch=999999
    return a


def operational_parent():return parent.config(parent.rows()[0])
