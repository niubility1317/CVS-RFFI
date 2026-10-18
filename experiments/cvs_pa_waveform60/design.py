"""Only the ordinary L PA return changes during the two existing E21..30 tails."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_noise_shape_factorial43 import design as old

ROOT = Path(__file__).resolve().parents[2]
PROJECT, SOURCE = old.PROJECT, old.SOURCE
RUN = '20261018-phase1-FcrossRX-PAwaveform-tail-e30-s2026092701-m2-r01'
RELEASE = 'cvs_FcrossRX_PA_waveform_tail_E30_20261018_r01'
OWNER = 'codex/root/PA-waveform60'
BASE = Path(PROJECT) / 'runs' / RUN
REFERENCE_RUN = old.RUN
SEED, STEPS, UPDATES = 2026092701, 222, 2220
RX = FIT = (1, 3, 4, 6, 8)
ARMS = ('NOAS_PAOFF', 'AS_PAOFF')
FULL_FP32_POLICY, CLASSES, SCENES, VIEWS = old.FULL_FP32_POLICY, old.CLASSES, old.SCENES, old.VIEWS
FORMAL_VIEWS, TRUTH, PANEL, SOURCE_CACHE = old.FORMAL_VIEWS, old.TRUTH, old.PANEL, old.SOURCE_CACHE
TRAIN_SCENES = ('practical_low_suburban', 'practical_mid_urban', 'practical_low_urban')
read, write = old.read, old.write

def rows():
    return [dict(arm=a, row_id=a + '-s' + str(SEED), model_seed=SEED) for a in ARMS]

def old_config(arm):
    if arm not in ARMS:
        raise ValueError('Only the two registered PA-waveform arms')
    previous = 'B01' if arm == 'AS_PAOFF' else 'B00'
    return old.config(next(r for r in old.rows() if r['arm'] == previous), 30)

def config(row, segment=30):
    if row not in rows() or segment != 30:
        raise ValueError('Only two own-parent E21..30 tails; no scratch/E50')
    inherited = old_config(row['arm'])
    c = deepcopy(inherited)
    c.update(schema='FcrossRX_PA_waveform60_v1', run_id=RUN, **row,
             launch_owner=OWNER, output_root=str(BASE / row['row_id'] / 'source'),
             parent_source=inherited['parent_source'],
             checkpoint_sources=[inherited['parent_source'] + '/final_ssdg.pth'],
             model_initialization='Exact corresponding P43 scratch20 student and own EMA; reset AdamW/cos10/cache',
             method='ordinary_L_PA_waveform_shadow_bypass_tail',
             stage='FcrossRX_fullsource_PA_waveform60',
             training_recipe='Original sixL128/1.25L/ORD128 SupCon/RDrop/oldU; only discard ordinary L PA waveform after full native shadow draw',
             PA_waveform=dict(scope='ordinary visible L128 only during E21..30',
                              parent_E1_E20_PA=True, native_no_pa=False,
                              native_p_pa_unchanged=True, frozen_schedule_epoch=13,
                              original_simulate_pa_runs_on_clone=True,
                              returned_waveform='exact non-PA prefix and zero PA strength',
                              full_Python_NumPy_Torch_RNG_consumption_retained=True,
                              injected_PA_packets=0, U_changed=False,
                              As_changed=False, raw_slots_changed=False),
             scene_prior=dict(scenes=list(TRAIN_SCENES), probabilities=[1 / 3] * 3,
                              constant=True, adaptive=False, state_filter=False,
                              receiver_lock_filter=False),
             baseline_reference='Reuse P43 B00/B01 FINAL30; each new arm has corresponding own P43 scratch20 parent; no reference retraining or rescoring',
             budget_authorization='Two new10 tails, each2220 updates; corresponding own20 parents already4440 updates; cumulative30 only',
             benchmark_context='Static ordinary-augmentation audit motivates PA waveform interaction ablation; PA label damage is not assumed',
             goal=dict(source_fullV_AS_clean_vs_new_NOAS_min_pp=-.1,
                       source_panel_AS_mid_low_mean_vs_originalAs30_min_pp=.5,
                       source_panel_AS_low_vs_originalAs30_min_pp=0.,
                       source_FAIL_still_all7=True, no_automatic_expansion=True))
    return c

def validate(c):
    row = next((r for r in rows() if r['row_id'] == c.get('row_id')), None)
    if row is None or c != config(row):
        raise ValueError('Fixed PA-waveform-only/all-source/default-head/sixL/oldU/E30 contract differs')
    return c

def make_args(c, device='cuda:0'):
    validate(c)
    a = old.make_args(old_config(c['arm']), device)
    a.run_id, a.candidate_id, a.output_dir = RUN, c['row_id'], c['output_root']
    a.baseline_ckpt = c['parent_source'] + '/final_ssdg.pth'
    return a
