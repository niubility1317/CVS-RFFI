"""Eight matched scratch E50 positions; a single source-only pilot precedes seeds."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_urban_sigtime50 import design as reference
from experiments.cvs_urban_sigtime import design as original

ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,FULL_FP32_POLICY=(getattr(reference,k) for k in
    ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','FULL_FP32_POLICY'))
RUN='20261015-phase1-sigtime-branch-align-e50-manysig-s2026092701-m8-r01'
RELEASE='cvs_sigtime_branch_align_e50_20261015_r01'
OWNER='codex/root/branch-align-e50-20261015'
BASE=Path(PROJECT)/'runs'/RUN
FORMAL_VIEWS,SCENES,VIEWS,URBAN,TRAIN_SCENES=(getattr(reference,k) for k in
    ('FORMAL_VIEWS','SCENES','VIEWS','URBAN','TRAIN_SCENES'))
SEEDS=(2026092701,)
FORMAL_SEEDS=(2026092701,2026092702,2026092703)
ARMS=('A0','A-T','A-F','A-P','A-R','A-S','A-J','A-ALL')
EPOCHS,LABEL_EPOCHS,PSEUDO_EPOCHS,STEPS,UPDATES=50,33,17,222,11100
WARMUP_EPOCHS=5
read,write=reference.read,reference.write


def rows():return [dict(arm=a,model_seed=s,row_id=f'{a}-s{s}') for s in SEEDS for a in ARMS]


def calibration_path():return BASE/f'A0-s{SEEDS[0]}'/'source'/'branch_calibration.json'


def config(row):
    if row not in rows():raise ValueError('Unregistered eight-position pilot row')
    c=deepcopy(reference.config(next(r for r in reference.rows() if r['arm']=='B0')))
    c.update(schema='sigtime_branch_alignment_e50_v1',run_id=RUN,**row,method='cvs_urban_branch50',
        launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),
        paired=False,ordinary_concat=True,curriculum=False,margin=False,urban_ssl=False,
        source_roles=dict(L_s=6300,L_train=5580,L_dev=720,U_s=56700,V=27000),
        internal_source=dict(fold_seed=2026101503,holdout_per_TX_RX_day=8,
            L_dev='read-only calibration and source development; never trained',V='health_only',
            old_full_L_checkpoint_initialization=False,historical_full_L_dev_exposure=True),
        auxiliary=dict(position=row['arm'][2:] if row['arm']!='A0' else 'none',
            kind='unit direction; eps=1e-4',warmup_epochs=WARMUP_EPOCHS,start_epoch=6,
            teacher='own scratch EMA clean eval stop-gradient',role='L_train only',
            reliability='common clean EMA default-head correct * urban scene; no urban student gate',
            teacher_forward_in_A0=True,head_aux_gradient=False,gain_aux_gradient=False,CE_updates_all=True,
            coefficients='once frozen source calibration from common A0 epoch5; identical for all positions and later seeds',
            calibration_path=calibration_path().as_posix(),ALL_weights={k:.2 for k in ('T','F','P','R','S')},
            other_consistency_losses=[]),
        comparison_baseline='A0 ordinary concatenation; same internal source fold, native old U gates, extra EMA forward and warmup',
        expansion='Single-seed eight-row pilot. All eight configs remain in the formal three-seed family; source-only family rules govern expansion, never target scores.',
        source_evidence='User requested complete same-loss position comparison; no branch filtering by frozen replacement or target results')
    c['seeds']['model']=row['model_seed']
    c['curriculum_plan']['advancement']='disabled: eta=0 nominal three-scene sampling throughout'
    c['margin_plan']['coefficient']=0.
    return c


def validate(c):
    row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
    if row is None or c!=config(row):raise ValueError('Fixed eight-position E50 pilot configuration differs')
    return c


def make_args(c,device='cuda:0'):
    validate(c)
    a=original.make_args(original.config(next(r for r in original.rows()
        if r['arm']=='B0' and r['model_seed']==c['model_seed'])),device)
    a.epochs=EPOCHS;a.label_epochs=LABEL_EPOCHS;a.pseudo_epochs=PSEUDO_EPOCHS
    a.output_dir=c['output_root'];a.run_id=RUN;a.candidate_id=c['row_id']
    return a


def recipe():
    value=read(calibration_path())
    if (value['status']!='SOURCE_BRANCH_CALIBRATED' or not value['source_only'] or value['query_read']
            or value['optimizer_updates'] or value['warmup_epoch']!=5 or value['feature_parameters']!=195041):
        raise ValueError('Need one valid global source calibration, not an older per-group recipe')
    return value
