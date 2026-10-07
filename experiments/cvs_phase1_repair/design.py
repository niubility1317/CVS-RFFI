import math
from pathlib import Path
from experiments.cvs_phase1_stack.design import (
    PROJECT, SOURCE, CAPSULE, TRUTH, SEEDS, CLASSES, FULL_FP32_POLICY, read, write,
    args_from_validated_config,
)

ROOT = Path(__file__).resolve().parents[2]
RUN = '20261008-phase1-reference-repair-manysig-m32-r01'
RELEASE = 'cvs_reference_repair_20261008_r01'
BASE = Path(PROJECT) / 'runs' / RUN
OWNER = 'codex/root/reference-repair-20261008'
ARMS = ('ce_cosine', 'leo_cosine', 'pseudo_batch_constant', 'pseudo_bank_constant',
        'pseudo_batch_cosine', 'pseudo_bank_cosine', 'bank_cosine_domain', 'bank_cosine_mixstyle')


def cosine_lr(base, epoch, total=200, minimum=1e-6):
    if not 1 <= epoch <= total or total < 2 or not 0 <= minimum <= base:
        raise ValueError('Invalid cosine schedule')
    return minimum + (base - minimum) * (1 + math.cos(math.pi * (epoch - 1) / (total - 1))) / 2


def rows():
    return [dict(stage='repair', arm=a, model_seed=s, row_id=a+'-s'+str(s)) for s in SEEDS for a in ARMS]


def config(row):
    arm = row['arm']
    f = [] if arm == 'ce_cosine' else ['leo']
    if arm not in ('ce_cosine', 'leo_cosine'):
        f += ['twostage', 'ema', 'pseudo']
    if arm == 'bank_cosine_domain': f += ['domain']
    if arm == 'bank_cosine_mixstyle': f += ['mixstyle']
    return dict(schema='reference_repair_v1', run_id=RUN, **row, features=sorted(f),
        variant='reference_response', source_contract=SOURCE, dataset=PROJECT+'/Dataset_WigSig/ManySig.pkl',
        output_root=(BASE/row['row_id']/'source').as_posix(), epochs=200, steps_per_epoch=222,
        model_initialization='scratch', checkpoint_sources=[], numerical_policy=FULL_FP32_POLICY,
        method='cvs_phase1_repair', target_access=False, pseudo_domain_gate=False,
        temporal_mode='epoch_bank' if 'bank' in arm else 'batch_neighbor', temporal_min_streak=2,
        lr_schedule='constant' if arm.endswith('constant') else 'cosine', lr_min=1e-6)


def validate(c):
    row = next((r for r in rows() if r['row_id'] == c.get('row_id')), None)
    if row is None or c != config(row): raise ValueError('Unregistered or modified repair row')
    return c


def make_args(c, device='cuda:0'):
    validate(c)
    a = args_from_validated_config(c, device)
    a.pseudo_domain_gate = False
    a.pseudo_temporal_mode = c['temporal_mode']
    a.pseudo_temporal_bank_min_streak = c['temporal_min_streak']
    a.phase1_lr_schedule = c['lr_schedule']; a.phase1_lr_min = c['lr_min']
    return a
