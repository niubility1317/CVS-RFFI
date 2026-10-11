"""Fixed report-aligned B0--B3 plan; no target feedback or adaptive sweep."""
import math
from pathlib import Path
from copy import deepcopy
from experiments.cvs_stage2_decouple import design as parent_design

ROOT = Path(__file__).resolve().parents[2]
PROJECT = parent_design.PROJECT
SOURCE = parent_design.SOURCE
CAPSULE = parent_design.CAPSULE
TRUTH = parent_design.TRUTH
CLASSES = parent_design.CLASSES
FULL_FP32_POLICY = parent_design.FULL_FP32_POLICY
read, write = parent_design.read, parent_design.write
SEEDS = parent_design.SEEDS
ARMS = ('B0', 'B1', 'B2', 'B3')
RUN = '20261011-phase1-stage2-geometry-manysig-m12-r01'
AUDIT_RUN = '20261011-phase1-stage2-geometry-source-manysig-m3-r01'
RELEASE = 'cvs_stage2_geometry_20261011_r01'
BASE = Path(PROJECT) / 'runs' / RUN
AUDIT_BASE = Path(PROJECT) / 'runs' / AUDIT_RUN
OWNER = 'codex/root/stage2-geometry-20261011'
MARGIN_REQUIRED = math.log(95.)


def parent_config(seed):
    return parent_design.config(next(r for r in parent_design.rows() if r['model_seed'] == seed and r['arm'] == 'C0'))


def rows():
    return [dict(arm=arm, model_seed=seed, row_id=f'{arm}-s{seed}') for seed in SEEDS for arm in ARMS]


def config(row):
    p = parent_config(row['model_seed'])
    return dict(schema='stage2_identifiable_geometry_v1', run_id=RUN, **row,
        method='cvs_stage2_geometry', variant='reference_response',
        dataset=p['dataset'], source_contract=SOURCE, parent_checkpoint=p['parent_checkpoint'],
        parent_prediction=p['parent_prediction'], checkpoint_sources=p['checkpoint_sources'],
        parent_contract=p, preparation=(AUDIT_BASE / f"s{row['model_seed']}").as_posix(),
        output_root=(BASE / row['row_id'] / 'source').as_posix(),
        epochs=40, steps_per_epoch=222, updates=8880, batch_size=128, label_smoothing=0.,
        optimizer=deepcopy(p['optimizer']), checkpoint_selection='final_only',
        losses=dict(keep=.1, finite_lt=1., receiver=1., margin_required=MARGIN_REQUIRED,
                    raw_logit_scale=30., ramp_steps=444, augmented_ce=False, ce_ratio_calibration=False),
        enabled=dict(keep=row['arm'] != 'B0', actions=row['arm'] != 'B0', receiver=row['arm'] == 'B3'),
        action_policy='none' if row['arm']=='B0' else 'coverage' if row['arm']=='B1' else 'conditional_mixture',
        action=dict(coverage_fraction=.5, shared_normalized_radius='Uniform(0,1)',
                    direction='unit Linfinity; common radius distribution, physical IQ effect measured separately',
                    fit_passes=2, fit_batch=128, online_fit=False),
        schedule=dict(action_every=4, receiver_every=8, action_batch=32, relation_batch=32,
                      action_order=['linear','linear','temporal','temporal']),
        receiver=dict(rank=2, ridge=.1, max_alpha=.25, vector_reliability=True,
                      complete_feature_dim=160, finite_population_variance_gradient=True,
                      block_potential_scaling=True, clip_negative_estimator=False),
        sampling='same private ordinary and uniform without-replacement group relation streams across arms',
        target_access=False, unlabeled=False, teacher=False, satellite=False, test_views=['clean'],
        numerical_policy=deepcopy(FULL_FP32_POLICY),
        evaluation=dict(primary_arm='B3',goal_three_seed_mean=.84,truth_last=True,feedback_to_training=False),
        concurrency=dict(max_processes_per_gpu=4,includes_existing=True), launch_owner=OWNER)


def validate(c):
    row = next((r for r in rows() if r['row_id'] == c.get('row_id')), None)
    if row is None or c != config(row):
        raise ValueError('Unregistered geometry configuration')
    return c
