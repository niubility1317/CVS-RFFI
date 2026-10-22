"""Frozen follow-up matrix. Planning/validation only; never launches a worker."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

from .plan import make_plan
from .gfactor_scope import main_source

SEEDS = (2026092701, 2026092702, 2026092703)
PREFIX = '20261022-phase1-'
CONTROLS = PREFIX + 'PAMF-completeness-controls-m15-r01'
GFACTOR = PREFIX + 'PAMF-PL-Gfactor-readouts-m51-r01'
EXTERNAL = PREFIX + 'DURA-external-Lonly-readouts-m24-r01'
GEOMETRY = PREFIX + 'PAMF-CRIC-source-physics-m15-r01'
TIMING = PREFIX + 'PAMF-completeness-timing-r01'
MAIN = '20261021-phase1-T1-DURA-comparison-m153-r01'
REPAIRED_F = PREFIX + 'T1-DURA-F-ablation-binding-m63-r01'
REPAIRED_CONTROLS = PREFIX + 'T1-DURA-F0-controls-binding-m6-r01'
EXTERNAL_PARENT = '20261019-phase1-external-dg-concat-pl-m219-r03'

STRUCTURES = {
    'SINC_FIXED_INIT': dict(
        change='Freeze the scratch-initial 24x79 Sinc kernels; keep the body and default head trainable.',
        control='The adaptive Sinc X3_CE F15 baseline; identical initial kernels, shared IQ filter, padding and scale.',
        limitation='48 fewer trainable frequency coordinates; inference kernel shape unchanged. Not a parameter-matched optimizer test.'),
    'FIR_SINC_INIT': dict(
        change='Replace Sinc synthesis by one unconstrained 24x1x79 real FIR bank shared across I/Q, bias=False.',
        control='Initialize from the same scratch Sinc kernels, not a trained checkpoint; all other initial tensors preserved.',
        limitation='1896 trainable FIR taps versus 48 Sinc frequency coordinates; ordinary AdamW units, no Hz multiplier. No equal-parameter claim.'),
    'PA_LINEAR_LIFT': dict(
        change='Replace every odd-order memory-polynomial lift block by its order1 delayed IQ, repeated to the original width.',
        control='Keep memory depth, clipping, envelope gate, convolutions, PA statistics, projections, head and all trainable tensors.',
        limitation='Equal trainable capacity but reduced input rank; tests the high-order lift only, not the entire PA branch or hardware identifiability.'),
}
PAIRINGS = {
    'ALL_RX_POS': 'same TX, any RX, different physical ID; exclude self and duplicate physical IDs',
    'WITHIN_RX_POS': 'same TX and same physical RX, different physical ID; exclude self and duplicate physical IDs',
}


def source_physics():
    return dict(role='L_s', U_IQ_reads=0, target_reads=0, fitted_probe_updates=0,
        samples_per_TX_RX_day_at_most=8, maximum_F0_samples=720,
        intervention_count=11, definition='experiments.cric_dura_comparison.source_physics',
        metrics=['joint160 cosine displacement', 'default-head flip fraction',
                 'reference descriptor relative L2', 'same-day anchor-balanced cross-RX identity geometry'],
        selection='The same physical IDs for all frozen readers; no score-dependent selection or parameter update')


def build_plan():
    original = make_plan()
    protocols = original['protocols']
    runs = {name: [] for name in (CONTROLS, GFACTOR, EXTERNAL, GEOMETRY, TIMING)}
    for seed in SEEDS:
        for arm, definition in STRUCTURES.items():
            runs[CONTROLS].append(dict(row_id=arm + '-s' + str(seed), arm=arm,
                protocol_id='F0', model_seed=seed, kind='new_F',
                regime='matched_sat_supervised', ablation='F15',
                U_IQ_reads=0, own_EMA=False, CRIC=False, RDrop=False,
                satellite_concat=True, satellite_CE_weight=.68, label_smoothing=0.,
                epochs=50, trajectory=[20, 10, 20], optimizer_updates=11100,
                labeled_batch=128, steps_per_epoch=222, G_roles='L',
                optimizer='AdamW lr2e-4 wd1e-4; three cosine segments to1e-6; clip5',
                definition=deepcopy(definition), baseline=dict(run_id=REPAIRED_F, row_id='F15-s' + str(seed)),
                readouts=['raw', 'DURA-L'], source_physics=source_physics()))
        for arm, positive_rule in PAIRINGS.items():
            runs[CONTROLS].append(dict(row_id=arm + '-s' + str(seed), arm=arm,
                protocol_id='F0', model_seed=seed, kind='new_F',
                regime='native_cric', method_id='pamf_t1_e50',
                epochs=50, trajectory=[20, 10, 20], optimizer_updates=11100,
                labeled_batch=128, steps_per_epoch=222, G_roles='LU',
                definition=dict(only_change='Labeled-source contrast positive-pair rule', positive_rule=positive_rule,
                    same_batch_and_features=True, temperature=.2, warm_epochs=5, ramp_epochs=10, final_weight=.1,
                    levels=['time_down', 't2.pool'], negatives='different TX only',
                    anchors='Average positive contributions per valid anchor, then valid anchors; log coverage and positive counts.',
                    empty_rule='finite differentiable zero; never invent physical RX or use U truth',
                    preserve='Native six L forwards, CE dose, As KL0, own EMA/U gates, entropy, default head and sampling'),
                baseline=main_source('F0', 'native_cric', seed),
                readouts=['raw', 'DURA-LU'], source_physics=source_physics()))
        for pid in protocols:
            runs[GFACTOR].append(dict(row_id=pid + '-PL-G-L-s' + str(seed),
                protocol_id=pid, model_seed=seed, kind='frozen_readout',
                source=main_source(pid, 'matched_sat_plain_pl', seed),
                source_regime='matched_sat_plain_pl', new_F_training=False, new_G_training=False,
                new_readout='DURA-L', reuse_raw=True, G_roles='L',
                existing_cells=['Sup + G-L', 'Sup + G-LU', 'PL + G-LU'],
                factorial='F noU/ordinaryPL x G L/LU, identical labels, views and source contract',
                warning='F has used U. This PL+G-L cell is not a fully L-only system.'))
        for family in ('riei', 'drift', 'ibrff', 'sdg'):
            for variant in ('ce', 'full'):
                source_row = 'F0-' + family + '-' + variant + '-s' + str(seed)
                runs[EXTERNAL].append(dict(row_id=source_row, protocol_id='F0',
                    model_seed=seed, kind='frozen_readout', method=family, variant=variant,
                    source=dict(run_id=EXTERNAL_PARENT, row_id=source_row),
                    source_regime='supervised', new_F_training=False, new_G_training=False,
                    reuse_raw=True, new_readout='DURA-L', G_roles='L',
                    policy='Same G-L weights, same fixed IQ and normalization; retain native model/head and training budget; no target adaptation.',
                    source_smoke='Load actual frozen checkpoint. On L-only IQ verify the unwrapped external model equals its native inference.',
                    comparison='External raw vs external+G-L; compare PAMF raw vs PAMF+G-L under the same permissions.',
                    limitation='Frozen common-preprocessing transfer, not a new DURA-trained external method.'))
        for arm in ('T1', 'F01', 'F02', 'F03', 'F16'):
            source = main_source('F0', 'native_cric', seed) if arm == 'T1' else dict(run_id=REPAIRED_F, row_id=arm + '-s' + str(seed))
            runs[GEOMETRY].append(dict(row_id=arm + '-s' + str(seed), protocol_id='F0',
                model_seed=seed, kind='frozen_source_diagnostic', source=source,
                new_F_training=False, new_G_training=False, new_target_predictions=False,
                source_physics=source_physics(),
                compare='Same fixed L IDs, joint160 geometry; contrast off/shallow/middle/final vs default shallow+middle.'))
    runs[TIMING].append(dict(row_id='three-new-structures', kind='exclusive_timing',
        protocol_id='F0', model_seed=SEEDS[0], arms=list(STRUCTURES),
        readers=['F', 'DURA-L+F'], batches=[1, 8, 32, 64, 128], rounds=5,
        case_rounds=150, precision='FP32/TF32off', source_role='L_s',
        GPU='Exclusive existing N607 GPU, excluding2/3; never interrupt healthy training.',
        reuse='Existing X3_CE timing baseline from supervised-physics panel; no uniform500 or earlier450 remeasurement.',
        metrics=['registered/deployed/trainable parameters separately', 'weight bytes', 'incremental memory',
                 'synchronized wall median/P95', 'throughput', 'load/capture overhead']))
    result = dict(schema='pamf_completeness_followup_v1', status='PLANNED_NOT_PUBLISHED',
        design_frozen_date='2026-10-22', protocols=protocols, model_seeds=list(SEEDS), runs=runs,
        scientific_scope='Existing Phase1 engineering benchmark; no new blind-test or in-orbit claim.',
        new_F_training=15, new_G_training=0, new_model_epochs=750, new_F_updates=166500,
        new_target_readout_rows=90, reused_F_readout_rows=75,
        new_source_only_rows=15, new_exclusive_case_rounds=150,
        total_new_stage_jobs=211,  #15F +90predict +90score +15source +1timing
        source_count_sweep_added=False, excluded_source_counts=[5],
        launch_policy='Implement and minimally verify after the current registered science/prediction scope closes; publish a new immutable run. No current queue/worker mutation.',
        stop_rule='Two identical technical failures stop new submissions. Preserve outputs and healthy workers. No target-driven changes or selective repeats.',
        implementation_status='Pending. This module constructs the plan only; it is not a training or publishing entry.',
        statistics=dict(unit='Three actual F model seeds; conditional on fixed G for each exact contract.',
            paired='Per-seed PL/Sup, G-LU/G-L, Full/CE and factorial interaction; all signs retained.',
            hierarchy='Per scenario/TX/RX/day, RX-equal and packet-weighted Accuracy/Macro-F1, clean damage and paired net corrections.',
            no_pseudoreplication='Packets, seven views, 5 timing rounds and 3x3 G/F cells are not independent model repetitions.'),
        reuse_only=['Existing F0 final T1 weights', 'All external source weights and raw decisions',
                   'Exact-contract G-L/G-LU full lineages', 'm3-natural three-G by three-F robustness',
                   'Completed external360 statistics, G-ablation scores and uniform500 latency'],
        not_required_for_current_claim=['More source RX permutations: current count trend is explicitly conditional on one fixed nested chain.',
            'Replicated copy/KD2x2: current G panel makes only supported simple-effect claims, no unobserved interaction.',
            'New Phase2/unknown/real-satellite trials: outside the current Phase1 comparison.',
            'Target-based tuning grids or a fully labeled U upper bound: not authorized by current data permissions.'])
    validate(result)
    return result


def validate(plan):
    if set(plan['runs']) != {CONTROLS, GFACTOR, EXTERNAL, GEOMETRY, TIMING}:
        raise ValueError('Unexpected supplement run set')
    expected = {CONTROLS: 15, GFACTOR: 51, EXTERNAL: 24, GEOMETRY: 15, TIMING: 1}
    if {k: len(v) for k, v in plan['runs'].items()} != expected:
        raise ValueError('Incomplete follow-up matrix')
    for run, rows in plan['runs'].items():
        if len({r['row_id'] for r in rows}) != len(rows):
            raise ValueError('Colliding immutable row IDs: ' + run)
    cells = Counter((r['protocol_id'], r['model_seed']) for r in plan['runs'][GFACTOR])
    if set(cells) != {(pid, seed) for pid in plan['protocols'] for seed in SEEDS} or any(v != 1 for v in cells.values()):
        raise ValueError('PL/G factor lacks a protocol/seed')
    if any(p.get('source_count') == 5 for p in plan['protocols'].values()):
        raise ValueError('User-excluded m5 source-count sweep')
    for r in plan['runs'][CONTROLS]:
        if r['optimizer_updates'] != 11100 or r['epochs'] != 50 or r['trajectory'] != [20, 10, 20]:
            raise ValueError('Unmatched F training budget')
        if r['arm'] in STRUCTURES and (r['U_IQ_reads'] != 0 or r['G_roles'] != 'L' or r['satellite_CE_weight'] != .68):
            raise ValueError('Strict supervised control widened permissions')
    if plan['new_G_training'] != 0 or plan['total_new_stage_jobs'] != 15 + 90 * 2 + 15 + 1:
        raise ValueError('Unexpected new training or stage count')


def audit_prediction_coverage(snapshot):
    """Input contains configs/statuses only, never IQ, scores or truth."""
    if snapshot.get('truth_or_score_reads') is not False:
        raise ValueError('The scope audit may not consume target scores')
    by_key = {}
    for row in snapshot['rows']:
        if row.get('scope') == 'main':
            key = (row['protocol_id'], row['regime'], row['model_seed'])
            if key in by_key:
                raise ValueError('Duplicated effective logical main prediction')
            by_key[key] = row
    expected = {(pid, regime, seed) for pid in make_plan()['protocols']
        for regime in ('native_cric', 'matched_sat_supervised', 'matched_sat_plain_pl') for seed in SEEDS}
    if set(by_key) != expected:
        raise ValueError('Effective main scope differs from17x3x3; inspect replacements before registering missing experiments')
    missing = []
    for key, row in sorted(by_key.items()):
        desired = {'raw', 'DURA-LU'}
        if row['regime'] != 'native_cric':
            desired.add('DURA-L')
        for name in sorted(desired - set(row['readouts'])):
            missing.append(dict(protocol_id=key[0], regime=key[1], model_seed=key[2], readout=name))
    return dict(status='AUDITED_CONFIGS_ONLY', main_logical_rows=len(by_key),
        missing_readout_cells=missing, missing_count=len(missing),
        current_owner=snapshot['current_owner'], current_stage_counts=snapshot['effective_counts'],
        truth_or_score_reads=False, no_training_or_runtime_mutations=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    target = Path(args.output)
    if target.exists():
        raise FileExistsError('Preserve existing frozen plan')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(build_plan(), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(status='PLANNED_NOT_PUBLISHED', new_F=15, new_G=0, target_rows=90, source_rows=15, stage_jobs=211)))
