"""Small balanced synthetic arrays only; no source/target artifact access."""
import copy
import itertools
import json
import math

import numpy as np
import pytest

from experiments.cvs_readout_attribution.attribution_statistics import (
    CONDITIONS, FACTORS, FEATURES, analyze_arrays,
)


@pytest.fixture
def arrays():
    # Orthogonal +/-1 contrasts: each factor has known SS = n * coefficient^2.
    levels = list(itertools.product(range(2), range(2), range(2), range(2)))
    code = np.asarray(levels)
    n = len(code)
    tx, rx, day, replicate = (2*code[:, k]-1 for k in range(4))
    effects = np.stack((tx, rx, day, tx*rx, tx*day, rx*day, tx*rx*day, replicate), axis=1)
    value = effects @ np.arange(1, 9, dtype=np.float64)
    features = np.zeros((n, 320), dtype=np.float64)
    features[:, 0] = value+13.5
    features[:, 1:] = .1
    truth = code[:, 0]
    logits = np.zeros((n, 6), dtype=np.float64)
    logits[np.arange(n), truth] = 3
    q = dict(ids=np.array([f'synthetic-{i}' for i in range(n)]), truth=truth,
             receiver=np.where(code[:, 1] == 0, 1, 3), day=code[:, 2]+1)
    q.update({key: logits.copy() for key in CONDITIONS})
    q.update({key: features.copy()*(index+1) for index, key in enumerate(FEATURES)})
    return q


def test_recovers_all_seven_factor_energies_and_within_cell(arrays):
    report = analyze_arrays(arrays, require_complete=False)
    n = len(arrays['truth'])
    assert report['cell_count'] == 8 and report['samples_per_cell'] == 2
    assert report['statistics_dtype'] == 'float64'
    for scale, name in enumerate(FEATURES, start=1):
        scatter = report['factor_scatter'][name]
        expected = n*scale**2*np.arange(1, 9)**2
        assert list(scatter['components']) == list(FACTORS)
        for key, energy in zip(FACTORS, expected):
            assert scatter['components'][key] == pytest.approx(energy)
            assert scatter['fractions'][key] == pytest.approx(energy/expected.sum())
        assert scatter['total'] == pytest.approx(expected.sum())
        assert scatter['reconstructed_total'] == pytest.approx(scatter['total'])
        assert scatter['reconstruction_relative_error'] < 1e-12
    json.dumps(report, allow_nan=False)


def test_factor_decomposition_is_translation_and_row_order_invariant(arrays):
    original = copy.deepcopy(arrays)
    before = analyze_arrays(arrays, False)
    assert all(np.array_equal(arrays[key], original[key]) for key in arrays)
    order = np.arange(len(arrays['truth']))[::-1]
    for key in arrays:
        arrays[key] = arrays[key][order].copy()
    for key in FEATURES:
        arrays[key] += 1000
    after = analyze_arrays(arrays, False)
    for key in FEATURES:
        assert after['factor_scatter'][key]['components'] == pytest.approx(before['factor_scatter'][key]['components'])


def test_constant_features_have_zero_total_and_undefined_proportions(arrays):
    for key in FEATURES:
        arrays[key].fill(.1)
    report = analyze_arrays(arrays, False)
    for scatter in report['factor_scatter'].values():
        assert scatter['total'] == 0
        assert scatter['reconstructed_total'] == 0
        assert all(v == 0 for v in scatter['components'].values())
        assert all(v is None for v in scatter['fractions'].values())
        assert scatter['reconstruction_relative_error'] is None


def _predicted_logits(predictions, margin):
    logits = np.zeros((len(predictions), 6), dtype=np.float64)
    logits[np.arange(len(predictions)), predictions] = margin
    return logits


def test_paired_help_hurt_prediction_changes_and_ce_direction(arrays):
    truth = arrays['truth']
    category = np.arange(len(truth)) % 4
    on = np.where(np.isin(category, [0, 2]), truth, (truth+1) % 6)
    off = np.where(np.isin(category, [1, 2]), truth, (truth+2) % 6)
    arrays['all_on'] = _predicted_logits(on, 4)
    arrays['both_off'] = _predicted_logits(off, 3)
    result = analyze_arrays(arrays, False)
    pair = result['paired_all_on']['both_off']
    assert pair['helped'] == pair['hurt'] == pair['both_correct'] == pair['both_wrong'] == 4
    assert pair['prediction_changed'] == 12  # Includes wrong-to-different-wrong.
    assert pair['accuracy_delta_pp'] == 0
    expected_ce_delta = 2+math.log1p(5*math.exp(-4))-(1.5+math.log1p(5*math.exp(-3)))
    assert pair['ce_delta'] == pytest.approx(expected_ce_delta)
    assert pair['ce_delta_summary']['mean'] == pair['ce_delta']
    assert result['classification']['all_on']['accuracy'] == .5
    assert result['classification']['all_on']['ce'] == pytest.approx(2+math.log1p(5*math.exp(-4)))
    assert len(result['classification']['all_on']['receiver']) == 2
    assert len(result['classification']['all_on']['cells']) == 8
    assert sum(c['count'] for c in result['classification']['all_on']['cells']) == len(truth)


@pytest.mark.parametrize('positive', [True, False])
def test_two_by_two_accuracy_interaction_sign(arrays, positive):
    truth = arrays['truth']
    correct = _predicted_logits(truth, 3)
    half_correct = _predicted_logits(np.where(np.arange(len(truth)) % 2, truth, (truth+1) % 6), 3)
    if positive:
        arrays.update(all_on=correct, time_off=half_correct, behavior_off=half_correct, both_off=half_correct)
    else:
        arrays.update(all_on=half_correct, time_off=correct, behavior_off=correct, both_off=correct)
    result = analyze_arrays(arrays, False)['accuracy_interaction']
    assert result['value_pp'] == (50 if positive else -50)
    assert all(c['value_pp'] == result['value_pp'] for c in result['cells'])
    assert all(c['value_pp'] == result['value_pp'] for c in result['receiver'].values())


def test_norm_ratios_use_all_samples_and_explicit_denominator_floor(arrays):
    for key in FEATURES:
        arrays[key].fill(0)
    arrays['time_skip'][:, 0] = 2
    arrays['time_delta'][:, 0] = np.arange(16)
    arrays['behavior_delta'][0, 0] = 1e-12
    result = analyze_arrays(arrays, False)['norm_ratios']
    assert result['time']['count'] == 16
    assert result['time']['median'] == 3.75
    assert result['time']['maximum'] == 7.5
    assert result['time']['p90'] == pytest.approx(6.75)
    assert result['behavior']['zero_skip_count'] == 16
    assert result['behavior']['floored_skip_count'] == 16
    assert result['behavior']['maximum'] == 1
    assert result['behavior']['mean'] == 1/16


def test_float32_input_and_large_common_logit_offset_are_stable(arrays):
    for key in FEATURES + CONDITIONS:
        arrays[key] = arrays[key].astype(np.float32)
    before = analyze_arrays(arrays, False)
    for key in CONDITIONS:
        arrays[key] = arrays[key].astype(np.float64)+100000
    after = analyze_arrays(arrays, False)
    for key in CONDITIONS:
        assert after['classification'][key]['ce'] == pytest.approx(before['classification'][key]['ce'])
    assert after['model_fitting'] is False and after['model_selection'] is False


@pytest.mark.parametrize('problem', ['missing_field', 'extra_field', 'bad_shape', 'logit_shape',
    'integer_feature', 'float_labels', 'label_range', 'metadata_shape', 'duplicate_id', 'empty_id',
    'nan_feature', 'inf_logits', 'missing_cell', 'unbalanced_cell'])
def test_rejects_invalid_or_unbalanced_arrays(arrays, problem):
    if problem == 'missing_field':
        arrays.pop('time_skip')
    elif problem == 'extra_field':
        arrays['target'] = np.zeros(1)
    elif problem == 'bad_shape':
        arrays['time_delta'] = arrays['time_delta'][:, :319]
    elif problem == 'logit_shape':
        arrays['all_on'] = arrays['all_on'][:, :5]
    elif problem == 'integer_feature':
        arrays['time_skip'] = arrays['time_skip'].astype(np.int64)
    elif problem == 'float_labels':
        arrays['receiver'] = arrays['receiver'].astype(np.float64)
    elif problem == 'label_range':
        arrays['truth'][0] = 6
    elif problem == 'metadata_shape':
        arrays['day'] = arrays['day'][:, None]
    elif problem == 'duplicate_id':
        arrays['ids'][0] = arrays['ids'][1]
    elif problem == 'empty_id':
        arrays['ids'][0] = ''
    elif problem == 'nan_feature':
        arrays['behavior_delta'][3, 4] = np.nan
    elif problem == 'inf_logits':
        arrays['time_off'][3, 4] = np.inf
    else:
        remove = 2 if problem == 'missing_cell' else 1
        arrays = {key: value[remove:] for key, value in arrays.items()}
    with pytest.raises(ValueError):
        analyze_arrays(arrays, False)


def test_complete_mode_rejects_small_balanced_fixture(arrays):
    with pytest.raises(ValueError, match='Complete source V'):
        analyze_arrays(arrays)


def test_empty_bytes_id_and_float64_overflow_are_rejected(arrays):
    arrays['ids'] = arrays['ids'].astype('S32')
    arrays['ids'][0] = b''
    with pytest.raises(ValueError, match='IDs'):
        analyze_arrays(arrays, False)
    arrays['ids'][0] = b'restored-id'
    arrays['all_on'][0, 0] = np.finfo(np.float64).max
    arrays['all_on'][0, 1] = -np.finfo(np.float64).max
    with pytest.raises(ValueError, match='float64 range'):
        analyze_arrays(arrays, False)


def test_complete_mode_accepts_exact_synthetic_source_grid():
    # A protocol-sized synthetic zero fixture, not a real source artifact.
    grid = np.asarray(list(itertools.product(range(6), (1, 3, 4, 6, 8), (1, 2, 3))))
    labels = np.repeat(grid, 300, axis=0)
    n = len(labels)
    q = dict(ids=np.arange(n), truth=labels[:, 0], receiver=labels[:, 1], day=labels[:, 2])
    features = np.zeros((n, 320), dtype=np.float32)
    q.update({name: features for name in FEATURES})
    q.update({name: np.zeros((n, 6), dtype=np.float32) for name in CONDITIONS})
    result = analyze_arrays(q)
    assert result['count'] == 27000 and result['cell_count'] == 90 and result['samples_per_cell'] == 300
    assert result['classification']['all_on']['accuracy'] == pytest.approx(1/6)
    assert result['classification']['all_on']['ce'] == pytest.approx(math.log(6))
    assert len(result['classification']['all_on']['cells']) == 90
    assert result['accuracy_interaction']['value_pp'] == 0
    q['receiver'] = np.where(q['receiver'] == 8, 7, q['receiver'])
    with pytest.raises(ValueError, match='Complete source V'):
        analyze_arrays(q)
