from copy import deepcopy
from pathlib import Path
from experiments.cvs_t1_e50 import design as old
from .plan import ARMS, SEED, VIEWS, profile

ROOT = Path(__file__).resolve().parents[2]
RUN = '20261021-phase1-SARD-DURAaug-e50-13L37U-s2026092701-m2-r01'
RELEASE = 'cvs_SARD_DURAaug_E50_13L37U_20261021_r01'
OWNER = 'codex/root/SARD-DURA-training'
PROJECT = Path(old.PROJECT)
BASE = PROJECT/'runs'/RUN
for key in ('SOURCE', 'FULL_FP32_POLICY', 'CLASSES', 'SCENES', 'FORMAL_VIEWS', 'TRUTH', 'RX', 'SOURCE_CACHE'):
    globals()[key] = deepcopy(getattr(old, key))
read, write = old.read, old.write
DURA_RELEASE = PROJECT/'releases/pamf_stage2_inference_20261021_r01'
DURA_COMMIT = '47c264286ca689f060a3f36aafb763fe209a6f06'
DURA_PARENT = PROJECT/'runs/20261021-phase1-restoration-R65-four-route-s2026092701-m4-r69'
DURA_CHECKPOINT = DURA_PARENT/'stage2/native_map_prior/final.pt'

def rows():
    return [dict(arm=arm, row_id=arm+'-s'+str(SEED), model_seed=SEED) for arm in ARMS]

def row_root(row):
    if row not in {r['row_id'] for r in rows()}:
        raise ValueError('Unregistered row')
    return BASE/row

def reference():
    return old.config(next(r for r in old.rows() if r['arm']=='SCRATCH50'))

def config(row):
    if row not in rows():
        raise ValueError('Unregistered scientific arm')
    c = deepcopy(reference())
    c.update(profile(row['arm']))
    c.update(schema='SARD_DURAaug_13L37U_v1', run_id=RUN, **row, launch_owner=OWNER,
             output_root=str(row_root(row['row_id'])/'source'), segment=20,
             parent_source=None, checkpoint_sources=[], model_initialization='scratch',
             native_scratch_only=True, method='SARD_with_frozen_DURA_source_augmentation',
             training_recipe='Original SARD sixL, ordinary/raw KL, As CE/KL0, ownEMA/oldUgate/CROSS_RX SupCon;13L37U',
             budget_authorization='User: scratch E50,13L+37U;single-seed matched augmentation falsification',
             source_selection='Fixed FINAL50;all rows auto raw+DURA seven-view independent truth-last',
             baseline_reference='Matched scratch SARD13L37U;historical SARD30+20 and CRIC descriptive only',
             test_views=list(VIEWS))
    c['clock'] = dict(scratch=True, label_epochs=13, pseudo_epochs=37, total_epochs=50,
        optimizer_resets=0, steps_per_epoch=222, total_steps=11100,
        LR='single50epoch cosine,unchanged native endpoints', historical_reference_calendar='metadata only')
    c['restored_augmentation'] = dict(enabled=row['arm']=='SARD_DURA_AUG', role='L_s',
        source='Same original practical random residual outputs,then fixedDURA', checkpoint=str(DURA_CHECKPOINT),
        runtime_commit=DURA_COMMIT, reuse=True, graph=True, native=False, precision='FP32/TF32off',
        G_updates=0, G_gradients=False, identity_labels_to_G=False, U_G_forwards=0,
        clean_branch='unchanged', As_KL=0., ordinary_raw_KL='unchanged', fallback=False)
    return c

def validate(c):
    row = next((r for r in rows() if r['row_id']==c.get('row_id')), None)
    if row is None or c != config(row):
        raise ValueError('Fixed scratch calendar,objective,source,seed or augmentation differs')
    return c

def make_args(c, device='cuda:0'):
    validate(c)
    a = old.make_args(reference(), device)
    a.run_id, a.candidate_id, a.output_dir = RUN, c['row_id'], c['output_root']
    a.epochs, a.label_epochs, a.pseudo_epochs = 50, 13, 37
    a.baseline_ckpt = a.teacher_ckpt = ''
    a.from_scratch = a.a1_scratch_only = True
    return a
