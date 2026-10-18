"""One independent-full-practical tail against the existing same-IQ AS_CE tail."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_satellite_dropout18 import design as old
from experiments.cvs_noise_shape_factorial43 import design as p43

ROOT = Path(__file__).resolve().parents[2]
PROJECT, SOURCE = old.PROJECT, old.SOURCE
RUN = '20261018-phase1-FcrossRX-fullAsPair6-e30-s2026092701-m1-r01'
RELEASE = 'cvs_FcrossRX_full_As_pair6_E30_20261018_r01'
OWNER = 'codex/root/full-As-pair68'
BASE = Path(PROJECT) / 'runs' / RUN
REFERENCE_RUN, SAME_RUN = p43.RUN, old.RUN
SEED, STEPS, UPDATES = 2026092701, 222, 2220
RX = FIT = (1, 3, 4, 6, 8)
ARMS = ('B01',)
LOCAL_NN, SMOKE_NN = 0, 772
FULL_FP32_POLICY, CLASSES, SCENES, VIEWS = old.FULL_FP32_POLICY, old.CLASSES, old.SCENES, old.VIEWS
FORMAL_VIEWS, TRUTH, PANEL, SOURCE_CACHE = old.FORMAL_VIEWS, old.TRUTH, old.PANEL, old.SOURCE_CACHE
TRAIN_SCENES = ('practical_low_suburban', 'practical_mid_urban', 'practical_low_urban')
read, write = old.read, old.write


def rows():
    return [dict(arm='B01', row_id='B01-s' + str(SEED), model_seed=SEED)]


def reference_config(arm='AS_CE'):
    if arm in ('AS_CE', 'SAME_CE6'):
        return old.config(next(r for r in old.rows() if r['arm'] == 'AS_CE'), 30)
    if arm in ('B00', 'B01'):
        return p43.config(next(r for r in p43.rows() if r['arm'] == arm), 30)
    raise ValueError('Only fixed SAT18 AS_CE or original P43 NoAs/As references')


def old_config(arm='B01'):
    if arm not in ARMS:
        raise ValueError('Only the independent full-As pair candidate')
    return reference_config('AS_CE')


def config(row, segment=30):
    if row not in rows() or segment != 30:
        raise ValueError('One existing AS_CE20 parent plus ten epochs only')
    inherited = old_config(row['arm'])
    c = deepcopy(inherited)
    c.update(schema='FcrossRX_full_channel_pair68_v1', run_id=RUN, **row,
        launch_owner=OWNER, output_root=str(BASE / row['row_id'] / 'source'),
        parent_source=inherited['parent_source'],
        checkpoint_sources=[inherited['parent_source'] + '/final_ssdg.pth'],
        model_initialization='Strict SAT18 AS_CE own20 student and ownEMA; reset AdamW/cos10/cache',
        method='INDEP_CE6', stage='FcrossRX_full_channel_pair68',
        training_recipe='Six L128/1.25 CE; original ordinary/raw KL and ORD128 SupCon/nativeU; two As64 CE slots have independent complete channels and no cross-channel KL',
        satellite_mode='AS_CE',
        full_as_pair=dict(method='INDEP_CE6', same_control='SAT18 AS_CE30',
            first_namespace='FcrossRX/train/e{epoch}/step{step}/L_S',
            second_namespace_suffix='/view1-full-independent',
            channel_seed=20261016421, receiver_seed=20261016423,
            hardware_session='rx{rx_i}:day{day_i}:eq{eq_i}',
            second_from_same_raw=True, second_from_first_As=False,
            exact_hardware_object_reused=True, two_separate_CE_mean=True,
            As_KL=False, ordinary_raw_KL=True, original_dropout=True,
            mixed_group_weight=.5, As_fraction=.5, As_CE_coefficient=.25,
            As_per_pass_CE_coefficient=.125, L_NN_per_step=768,
            generated_As_per_step=128, As_NN_per_step=128,
            extra_NN=0, extra_parameters=0, extra_losses=0,
            independent_given_raw_and_RX_hardware=True),
        scene_prior=dict(scenes=list(TRAIN_SCENES), probabilities=[1/3]*3,
            adaptive=False, state_filter=False, receiver_lock_filter=False),
        baseline_reference='Reuse SAT18 AS_CE30 exact ownAS_CE20 parent/sixL/loss control; P43 originalAs30/NoAs30 recipe references only',
        baseline_reuse=False,
        history_boundary='Old practicalMC2 trained independent complete views with ten L forwards and held RX1; source FAIL. This tests two CE-only As slots at fixed sixL/fullsource budget, not a new sampling algorithm.',
        parent_history_difference='SAT18 parent20 already has AsKLoff; original P43 As20 has AsKL. Do not attribute whole P43 differences only to the new ten-epoch pairing.',
        budget_authorization='One ownAS_CE20+10 total30/6660updates; existing SAME/P43 references reused; no automatic seed or E50 expansion',
        cost_claim='New10:2220updates,1704960LNN,128AsNN and128fullchannel generations perstep; no extra NN; extra channel CPU work measured',
        goal=dict(fullV_clean_vs_originalNoAs30_min_pp=-.1,
            pooled_midlow_vs_max_originalAs30_SAME_min_pp=.5,
            E50_only_after_source_PASS=True, E50_total_not_additional=True,
            source_FAIL_still_all7=True, target_feedback=False))
    return c


def validate(c):
    row = next((r for r in rows() if r['row_id'] == c.get('row_id')), None)
    if row is None or c != config(row):
        raise ValueError('Fixed ownAS_CE20/fullsource/sixL/independent fullchannel contract differs')
    return c


def make_args(c, device='cuda:0'):
    validate(c)
    a = old.make_args(old_config(c['arm']), device)
    a.run_id, a.candidate_id, a.output_dir = RUN, c['row_id'], c['output_root']
    a.baseline_ckpt = c['parent_source'] + '/final_ssdg.pth'
    if a.teacher_ckpt or a.pseudo_temporal_mode != 'batch_neighbor':
        raise ValueError('Original ownEMA/rawU/batch-neighbor rules must remain')
    return a
