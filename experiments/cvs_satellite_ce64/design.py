"""Satellite-only training CE temperature versus an explicit prefactor control."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_noise_shape_factorial43 import design as old

ROOT = Path(__file__).resolve().parents[2]
PROJECT, SOURCE = old.PROJECT, old.SOURCE
RUN = '20261018-phase1-FcrossRX-satelliteCE-temperature-e30-s2026092701-m2-r01'
RELEASE = 'cvs_FcrossRX_satellite_CE_temperature_E30_20261018_r01'
OWNER = 'codex/root/satellite-ce64'
BASE = Path(PROJECT) / 'runs' / RUN
REFERENCE_RUN = old.RUN
SEED, STEPS, UPDATES = 2026092701, 222, 2220
RX = FIT = (1, 3, 4, 6, 8)
ARMS = ('CE_HALF', 'CE_T2')
FULL_FP32_POLICY, CLASSES, SCENES, VIEWS = old.FULL_FP32_POLICY, old.CLASSES, old.SCENES, old.VIEWS
FORMAL_VIEWS, TRUTH, PANEL, SOURCE_CACHE = old.FORMAL_VIEWS, old.TRUTH, old.PANEL, old.SOURCE_CACHE
TRAIN_SCENES = ('practical_low_suburban', 'practical_mid_urban', 'practical_low_urban')
read, write = old.read, old.write

def rows():
    return [dict(arm=a, row_id=a + '-s' + str(SEED), model_seed=SEED) for a in ARMS]

def old_config(arm='CE_T2'):
    if arm not in ARMS:
        raise ValueError('Only registered satellite CE objective arms')
    return old.config(next(r for r in old.rows() if r['arm'] == 'B01'), 30)

def config(row, segment=30):
    if row not in rows() or segment != 30:
        raise ValueError('Only own As20 to E30 tail; no scratch/E50')
    inherited = old_config(row['arm'])
    c = deepcopy(inherited)
    c.update(schema='FcrossRX_satellite_ce64_v1', run_id=RUN, **row,
             launch_owner=OWNER, output_root=str(BASE / row['row_id'] / 'source'),
             parent_source=inherited['parent_source'],
             checkpoint_sources=[inherited['parent_source'] + '/final_ssdg.pth'],
             model_initialization='Exact P43 B01 As scratch20 student and own EMA; reset AdamW/cos10/cache',
             method='nominal_As_satellite_CE_temperature_tail',
             stage='FcrossRX_fullsource_satellite_ce64',
             training_recipe='Original sixL128/1.25L/ORD128 SupCon/RDrop/oldU; only satellite64 two-pass true-label CE changes',
             ce_objective=dict(temperature=2. if row['arm']=='CE_T2' else 1.,
                               weight=1. if row['arm']=='CE_T2' else .5, smoothing=.01),
             ce_objective_notes=dict(T_squared=False, deployment_head_scale=30.,
                       formula='delta=.25*(mean_two_pass_new_CE64-mean_two_pass_original_CE64)',
                       unchanged='ordinary/raw CE and symmetric RDrop KL; all original IQ and U/EMA/labels',
                       all_labeled_As_retained=True, extra_NN=0, extra_parameters=0,
                       control='same explicit 1/2 logit chain prefactor; not matched actual gradient norm',
                       hypothesis='temperature reshapes correct/uncertain packet gradients beyond uniform CE attenuation',
                       unsupported_hypothesis='source wrong urban packets are mostly uncertain, not high-confidence errors'),
             scene_prior=dict(scenes=list(TRAIN_SCENES), probabilities=[1 / 3] * 3,
                              constant=True, adaptive=False, state_filter=False, receiver_lock_filter=False),
             baseline_reference='Reuse P43 B00/B01 FINAL30; no reference retraining, prediction or scoring',
             budget_authorization='Two independent new10 tails/2220 updates each from same own20/4440; cumulative30 each',
             benchmark_context='New fixed satellite-only CE objective shape test against explicit prefactor and original fullCE; source-only choice before formal',
             goal=dict(source_fullV_AS_clean_vs_originalNOAS_min_pp=-.1,
                       source_panel_AS_mid_low_mean_vs_originalAs30_min_pp=.5,
                       source_panel_AS_low_vs_originalAs30_min_pp=0.,
                       source_T2_midlow_vs_HALF_min_pp=.5, source_FAIL_still_all7=True, no_automatic_expansion=True))
    return c

def validate(c):
    row = next((r for r in rows() if r['row_id'] == c.get('row_id')), None)
    if row is None or c != config(row):
        raise ValueError('Fixed all-source/default-head/sixL/nominalAs/oldU/CE64 E30 contract differs')
    return c

def make_args(c, device='cuda:0'):
    validate(c)
    a = old.make_args(old_config(c['arm']), device)
    a.run_id, a.candidate_id, a.output_dir = RUN, c['row_id'], c['output_root']
    a.baseline_ckpt = c['parent_source'] + '/final_ssdg.pth'
    return a
