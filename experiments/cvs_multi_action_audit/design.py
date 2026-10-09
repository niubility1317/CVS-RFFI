"""Fixed source diagnostics; no target access, candidate selection or identity update."""
from pathlib import Path
from experiments.cvs_multi_disentangle import design as parent
from experiments.cvs_phase1_overlay.contract import read, write

ROOT = Path(__file__).resolve().parents[2]
PROJECT = parent.PROJECT
RUN = '20261009-phase1-multi-action-audit-manysig-m4-r01'
RELEASE = 'cvs_multi_action_audit_20261009_r01'
BASE = Path(PROJECT)/'runs'/RUN
OWNER = 'codex/root/multi-action-audit-20261009'
SOURCE = parent.SOURCE
SEEDS = parent.SEEDS
EXPOSURES = ('clean', 'source_practical_mid')
RECIPE = dict(steps=200, batch_size=32, learning_rate=.0002, rank=8, code_dim=8,
    fit_per_tx_rx=64, audit_per_tx_rx=32, physical_split_seed=20261009,
    augmentation_seed=2026100901, evaluation_seed=2026100902,
    source_receivers=[1,3,4,6,8], source_classes=6, exposures=list(EXPOSURES),
    fit_modes=['legacy_self','block_z_self','block_z_cross'],
    identity_update=False, target_access=False, early_stopping=False,
    teacher='fixed_parent_multi_E200_student', parameter_weight=1., alpha_z=1.)

def rows():
    return [dict(row_id='audit-s'+str(s), model_seed=s) for s in SEEDS]

def parent_config(seed):
    return parent.config(next(r for r in parent.rows() if r['arm']=='multi' and r['model_seed']==seed))

def config(row):
    p=parent_config(row['model_seed'])
    return dict(schema='multi_action_audit_v1',run_id=RUN,**row,recipe=RECIPE,
        checkpoint=str(Path(p['output_root'])/'final_ssdg.pth'),parent_config=p,
        source_contract=SOURCE,output_root=str(BASE/row['row_id']),
        method='source_action_audit',target_access=False)

def validate(c):
    r=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
    if r is None or c!=config(r): raise ValueError('Unregistered diagnostic row')
    return c
