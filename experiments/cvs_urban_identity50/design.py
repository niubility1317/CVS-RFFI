"""Two scratch E50 rows; coefficients and augmentation fixed from source A only."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_urban_sigtime50 import design as reference

ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,FULL_FP32_POLICY=(getattr(reference,k) for k in
    ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','FULL_FP32_POLICY'))
RUN='20261015-phase1-sigtime-joint160-kl-dir-e50-manysig-s2026092701-m2-r01'
RELEASE='cvs_sigtime_joint160_kl_dir_e50_20261015_r01'
OWNER='codex/root/joint160-kl-dir-e50-20261015'
BASE=Path(PROJECT)/'runs'/RUN
FORMAL_VIEWS,SCENES,VIEWS,URBAN,TRAIN_SCENES=(getattr(reference,k) for k in
    ('FORMAL_VIEWS','SCENES','VIEWS','URBAN','TRAIN_SCENES'))
SEEDS=(2026092701,)
ARMS=('OUTPUT_KL','JOINT_DIR')
PRIMARY='JOINT_DIR'
EPOCHS,LABEL_EPOCHS,PSEUDO_EPOCHS,STEPS,UPDATES=50,33,17,222,11100
read,write=reference.read,reference.write


def rows():return [dict(arm=a,model_seed=s,row_id=f'{a}-s{s}') for s in SEEDS for a in ARMS]


def recipe():
    value=read(ROOT/'experiments/cvs_urban_identity50/source_recipe.json')
    if not value['source_only'] or value['query_read'] or value['added_training_updates']:
        raise ValueError('Source-only static coefficients required')
    if len(value['common_training_eta'])!=50 or any(not 0<=v<=.5 for v in value['common_training_eta']):
        raise ValueError('Fixed source A curriculum differs')
    return value


def config(row):
    if row not in rows():raise ValueError('Unregistered joint160 row')
    c=deepcopy(reference.config(next(r for r in reference.rows() if r['arm']=='C3')))
    p=recipe();kind='output_kl' if row['arm']=='OUTPUT_KL' else 'joint_direction'
    c.update(schema='sigtime_joint160_distill_e50_v1',run_id=RUN,**row,method='cvs_urban_identity50',
        launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),
        margin=False,urban_ssl=False,ordinary_concat=False,paired=True,
        expansion='Two scratch single-seed rows only; no automatic expansion',
        comparison_baseline='Existing C3 E50 A; identical source A realized curriculum. B vs C isolates auxiliary target.',
        auxiliary=dict(kind=kind,coefficient=p['coefficient'][kind]['coefficient'],
            teacher='own scratch EMA clean, eval and stop gradient',role='L_s only',
            reliability='clean EMA default-head correct; urban scenes only; no student qualification',
            feature='actual deployed joint160, forward-identical detached-gain reconstruction',
            head_aux_gradient=False,gain_aux_gradient=False,CE_updates_all=True,
            other_consistency_losses=[],dropout='native student train / EMA eval; unchanged',
            source_calibration_run=p['run_id'],source_calibration_commit=p['commit']),
        frozen_eta=p['common_training_eta'],
        source_evidence='Static lambda and common eta from zero-update legal L_s calibration; no target scores consumed')
    c['curriculum_plan']['advancement']='Replay fixed source-only A schedule, identical for both rows; no online change'
    c['margin_plan']['coefficient']=0.
    return c


def validate(c):
    row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
    if row is None or c!=config(row):raise ValueError('Frozen joint160 E50 configuration differs')
    return c


def make_args(c,device='cuda:0'):
    validate(c);old=reference.config(next(r for r in reference.rows() if r['arm']=='C3'))
    a=reference.make_args(old,device)
    a.output_dir=c['output_root'];a.run_id=RUN;a.candidate_id=c['row_id']
    return a
