"""Corresponding P43 own20 moments, with the same new cosine10 and all data/losses."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_noise_shape_factorial43 import design as old

ROOT = Path(__file__).resolve().parents[2]
PROJECT, SOURCE = old.PROJECT, old.SOURCE
RUN = '20261018-phase1-FcrossRX-optimizerKeep-e30-s2026092701-m2-r01'
RELEASE = 'cvs_FcrossRX_optimizer_keep_E30_20261018_r01'
OWNER = 'codex/root/optimizer-keep69'
BASE = Path(PROJECT) / 'runs' / RUN
REFERENCE_RUN = old.RUN
SEED, STEPS, UPDATES = 2026092701, 222, 2220
RX = FIT = (1, 3, 4, 6, 8)
ARMS = ('B00', 'B01')
LOCAL_NN, SMOKE_NN = 0, 8
FULL_FP32_POLICY, CLASSES, SCENES, VIEWS = old.FULL_FP32_POLICY, old.CLASSES, old.SCENES, old.VIEWS
FORMAL_VIEWS, TRUTH, PANEL, SOURCE_CACHE = old.FORMAL_VIEWS, old.TRUTH, old.PANEL, old.SOURCE_CACHE
read, write = old.read, old.write


def reference_config(arm='B01'):
    if arm not in ARMS:
        raise ValueError('Only corresponding original P43 NoAs/As RESET references')
    return old.config(next(r for r in old.rows() if r['arm'] == arm), 30)


def old_config(arm='B01'):
    return reference_config(arm)


def rows():
    return [dict(arm=a, row_id=a + '-s' + str(SEED), model_seed=SEED) for a in ARMS]


def config(row, segment=30):
    if row not in rows() or segment != 30:
        raise ValueError('Two matching own20+10 tails only; E50 is a separate later run')
    inherited = old_config(row['arm'])
    c = deepcopy(inherited)
    c.update(schema='FcrossRX_optimizer_continuity69_v1', run_id=RUN, **row,
        launch_owner=OWNER, output_root=str(BASE / row['row_id'] / 'source'),
        parent_source=inherited['parent_source'], checkpoint_sources=[inherited['parent_source'] + '/final_ssdg.pth'],
        model_initialization='Strict corresponding P43 own20 student/ownEMA/AdamW state; new cosine10 and segment counters',
        method='AS_KEEP' if row['arm'] == 'B01' else 'NOAS_KEEP',
        stage='FcrossRX_fullsource_optimizer_continuity69',
        training_recipe='Original sixL768/1.25 CE+RDrop/ORD128 SupCon/fullAs64-or-ordinary/nativeU batch_neighbor; retain corresponding parent AdamW moments only',
        optimizer_policy='KEEP parent per-parameter exp_avg/exp_avg_sq/step; fresh parameter-group hyperparameters/cos10/CENTER_UNITS hooks and local counters',
        optimizer_keep=dict(checkpoint_field='optimizer', parent_updates=4440,
            retained=['exp_avg', 'exp_avg_sq', 'step'], old_LR_loaded=False,
            fresh_hyperparameters_restored=True, current_parameter_pointers_retained=True,
            parent_unused_parameters_may_have_no_state=True, require_every_parameter_step4440=False,
            fresh_Sinc_segment_counters=True, scheduler='same native cosine10',
            extra_NN=0, extra_losses=0, extra_parameters=0, extra_U_rules=0),
        baseline_reference='Reuse exact corresponding P43 own20+RESET10 FINAL30; two matched NoAs/As KEEP tails',
        cost_claim='Per new tail2220updates/native3162962NN,source12600NN,formal1176002NN;shared parent student/ownEMA smoke8NN;no extra training forward',
        budget_authorization='Two corresponding own20+10 total30/6660;sourcePASS may allow later separately registered total50;no seed scan',
        benchmark_context='Existing exposed benchmark; optimizer continuity is source-selected and formal scores never feed promotion',
        goal=dict(fullV_clean_AS_KEEP_vs_NOAS_KEEP_min_pp=-.1,
            pooled_midlow_AS_KEEP_vs_originalAS30_min_pp=.5,
            source_FAIL_still_all7=True, E50_only_after_source_PASS=True,
            E50_as_separate_run=True, no_automatic_seed_extension=True, target_feedback=False))
    return c


def validate(c):
    row = next((r for r in rows() if r['row_id'] == c.get('row_id')), None)
    if row is None or c != config(row):
        raise ValueError('Fixed matching-parent/fullsource/sixL/AdamW-KEEP total30 contract differs')
    return c


def make_args(c, device='cuda:0'):
    validate(c)
    a = old.make_args(old_config(c['arm']), device)
    a.run_id, a.candidate_id, a.output_dir = RUN, c['row_id'], c['output_root']
    a.baseline_ckpt = c['parent_source'] + '/final_ssdg.pth'
    if a.teacher_ckpt or a.pseudo_temporal_mode != 'batch_neighbor' or not a.pseudo_temporal_gate:
        raise ValueError('Native ownEMA and original temporal U rules must remain')
    return a
