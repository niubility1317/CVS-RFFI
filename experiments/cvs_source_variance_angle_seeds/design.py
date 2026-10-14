import copy
from pathlib import Path
from experiments.cvs_source_variance_angle import design as first
from experiments.cvs_backbone_fishr_seeds import design as models

ROOT = Path(__file__).resolve().parents[2]
PROJECT, SOURCE, FULL_FP32_POLICY, read, write = (
    getattr(first, k) for k in ('PROJECT', 'SOURCE', 'FULL_FP32_POLICY', 'read', 'write'))
RUN = '20261015-phase1-source-backbone-variance-angle-manysig-s0203-m2-r01'
RELEASE = 'cvs_source_variance_angle_seeds_20261015_r01'
OWNER = 'codex/root/source-variance-angle-seeds-20261015'
BASE = Path(PROJECT) / 'runs' / RUN
SEEDS = (2026092702, 2026092703)
PARAMETERS, COHORTS, EPS, QUALIFICATION = (
    getattr(first, k) for k in ('PARAMETERS', 'COHORTS', 'EPS', 'QUALIFICATION'))
SOURCE_DECISION_COMMIT = 'cc2409be311c416c7cf4ed3d0ea6409b79f3f973'
PARENT_EXECUTION = '3c052b2d36903e28694674a39e232a73acd8b6a6'


def rows():
    return [dict(row_id=f'SIG_TIME-s{s}', arm='SIG_TIME', model_seed=s) for s in SEEDS]


def source_config(row):
    if row not in rows():
        raise ValueError('Unregistered remaining source model')
    return models.config(next(r for r in models.rows() if r == row))


def config(row):
    if row not in rows():
        raise ValueError('Unregistered remaining source diagnostic')
    c = copy.deepcopy(first.config(first.rows()[0]))
    c.update(schema='source_backbone_variance_angle_remaining_v1', run_id=RUN, **row,
             source_config=source_config(row), checkpoint_parent_run=models.RUN,
             checkpoint_parent_execution=PARENT_EXECUTION,
             source_qualification_run=first.RUN, source_decision_commit=SOURCE_DECISION_COMMIT,
             first_seed_remeasured=False, combined_model_seeds=[2026092701, *SEEDS],
             output_root=(BASE / row['row_id']).as_posix(), launch_owner=OWNER)
    return c


def validate(c):
    row = next((r for r in rows() if r['row_id'] == c.get('row_id')), None)
    if row is None or c != config(row):
        raise ValueError('Fixed remaining-seed source-angle diagnostic differs')
