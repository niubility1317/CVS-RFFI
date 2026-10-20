"""Keep the original T1 objective; compare continuation and scratch E50."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_as_temperature84 import design as old
from .plan import ARMS, profile

ROOT = Path(__file__).resolve().parents[2]
RUN = '20261020-phase1-T1-AsKL0-e50-s2026092701-m2-r01'
RELEASE = 'cvs_T1_AsKL0_E50_20261020_r01'
OWNER = 'codex/root/T1-e50'
PROJECT = old.PROJECT
BASE = Path(PROJECT) / 'runs' / RUN
PARENT_ROW = next(r for r in old.rows() if r['arm'] == 'T1')
PARENT_CONFIG = old.config(PARENT_ROW)
PARENT_SOURCE = Path(PARENT_CONFIG['output_root'])
for _key in ('SOURCE', 'FULL_FP32_POLICY', 'CLASSES', 'SCENES', 'FORMAL_VIEWS', 'TRUTH', 'RX', 'SOURCE_CACHE'):
    globals()[_key] = deepcopy(getattr(old, _key))
from .evaluation_plan import VIEWS
read, write = old.read, old.write
SEED = 2026092701
STEPS = 222
GPU_IDS = tuple(range(8))
MAX_TOTAL_GPU_PROCESSES = 3


def rows():
    return [dict(arm=a, row_id=a + '-s' + str(SEED), model_seed=SEED) for a in ARMS]


def row_root(row):
    return BASE / row


def reference_config(segment):
    return old.reference_config(segment)


def config(row):
    if row not in rows():
        raise ValueError('Only registered T1 rows')
    p = profile(row['arm'])
    tail = row['arm'] == 'E30PLUS20'
    c = deepcopy(old.config(PARENT_ROW, 30 if tail else 20))
    c.update(p)
    c.update(schema='T1_E50_v1', run_id=RUN, **row, launch_owner=OWNER,
             output_root=str(row_root(row['row_id']) / 'source'),
             segment=30 if tail else 20, parent_source=str(PARENT_SOURCE) if tail else None,
             checkpoint_sources=[str(PARENT_SOURCE / 'final_ssdg.pth')] if tail else [],
             model_initialization='original_T1_FINAL30_student_and_own_EMA' if tail else 'scratch',
             native_scratch_only=not tail, training_recipe='Original T1 sixL AsCE/KL0; ordinary/raw RDrop and original hiddenU/ownEMA retained',
             budget_authorization='User requests both E30+20 and scratch50 at model seed2026092701',
             source_selection='Fixed FINAL50; no epoch selection; automatic clean-only truth-last',
             baseline_reference='Original T1 E30; paths have different label/U/LR calendars; no matched-calendar causal claim')
    c['training_mechanism'].update(expected_L_forward_packets=p['updates'] * 768,
                                   extra_L_identity_forwards=p['updates'] * 640)
    c['temperature'] = deepcopy(PARENT_CONFIG['temperature'])
    return c


def validate(c):
    row = next((r for r in rows() if r['row_id'] == c.get('row_id')), None)
    if row is None or c != config(row):
        raise ValueError('Fixed T1 budget/source/head/config differs')
    return c


def make_args(c, device='cuda:0'):
    validate(c)
    a = old.make_args(old.config(PARENT_ROW, c['segment']), device)
    a.run_id = RUN
    a.candidate_id = c['row_id']
    a.output_dir = c['output_root']
    a.epochs = c['epochs']
    a.label_epochs = c['label_epochs']
    a.pseudo_epochs = c['pseudo_epochs']
    a.from_scratch = c['segment'] == 20
    a.a1_scratch_only = a.from_scratch
    a.baseline_ckpt = '' if a.from_scratch else str(PARENT_SOURCE / 'final_ssdg.pth')
    a.teacher_ckpt = ''
    return a
