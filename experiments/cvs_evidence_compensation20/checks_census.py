"""Synthetic frozen-census checks; no IQ, checkpoint or model is opened."""
import copy
import json
import math
import random

import numpy as np
import torch

from .census import Census, COEFFICIENT, PI, SOURCE_DAYS, SOURCE_RX, VIEWS


def _fixture(count, offset=0):
    receivers = [SOURCE_RX[index % 5] for index in range(count)]
    days = [SOURCE_DAYS[(index // 5) % 3] for index in range(count)]
    ids = [(0, receiver, day, offset + index, offset + index)
           for index, (receiver, day) in enumerate(zip(receivers, days))]
    return dict(physical_ids=ids, rx=torch.tensor(receivers), day=torch.tensor(days),
                targets=torch.zeros(count, dtype=torch.long),
                accept_mask=torch.ones(count, dtype=torch.bool),
                target_probabilities={view: torch.full((count,), .99, dtype=torch.float64) for view in VIEWS})


def _equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and set(left) == set(right) and all(_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right))
    return left == right


def _rng():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def _close(observed, expected):
    assert math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-15), (observed, expected)


def check():
    rng = _rng()
    negative_count = 0

    def reject(census, **data):
        nonlocal negative_count
        state = copy.deepcopy(census.__dict__)
        try:
            census.add(**data)
        except ValueError:
            negative_count += 1
        else:
            raise AssertionError('Invalid frozen census input was accepted')
        assert _equal(state, census.__dict__), 'Rejected add changed state/counts/ID reservations'

    # 99% accepted teacher precision can coexist with 0% EVIDENCE-active precision.
    data = _fixture(100)
    data['source_truth'] = torch.zeros(100, dtype=torch.long)
    data['source_truth'][-1] = 1
    data['target_probabilities']['PA_WEAK'][-1] = .1
    data['target_probabilities']['REF_WEAK'][-1] = .2
    before = copy.deepcopy(data)
    census = Census('V_proxy'); census.add(**data)
    result = census.finish(100); row = result['summary']
    assert _equal(before, data), 'Census changed input decisions/probabilities/truth'
    _close(row['teacher_precision_among_q'], .99)
    assert row['EVIDENCE_active_count'] == 1 and row['new_active_vs_SELF_count'] == 1
    assert row['EVIDENCE_active_precision'] == 0.
    assert row['EVIDENCE_active_correct_count'] == 0 and row['EVIDENCE_active_error_count'] == 1
    receiver = str(int(data['rx'][-1]))
    assert result['by_RX'][receiver]['EVIDENCE_active_precision'] == 0.
    assert result['by_RX'][receiver]['EVIDENCE_active_error_count'] == 1
    assert len(result['strata']) == 15 and set(result['by_RX']) == {'1', '3', '4', '6', '8'}
    assert sum(value['sample_count'] for value in result['strata']) == 100
    assert sum(value['EVIDENCE_active_error_count'] for value in result['strata']) == 1
    expected = COEFFICIENT * (math.log(PI / .1) + math.log(PI / .2)) / 200
    _close(row['EVIDENCE_STOP_loss'], expected)
    assert row['SELF_STOP_loss'] == 0.
    assert result['truth_status'] == 'PROVIDED'
    # Returned summaries are detached scalar JSON and cannot change stored results.
    json.dumps(result, allow_nan=False)
    result['summary']['accepted_count'] = -1
    assert census.finish(100)['summary']['accepted_count'] == 100
    try:
        census.add(**_fixture(1, 100))
    except ValueError:
        negative_count += 1
    else:
        raise AssertionError('Finished census accepted another batch')

    # q=False suppresses challenge activity and loss without making a new decision.
    excluded = copy.deepcopy(data); excluded['accept_mask'][-1] = False
    q_census = Census('V_proxy'); q_census.add(**excluded)
    q_row = q_census.finish(100)['summary']
    assert q_row['accepted_count'] == 99 and q_row['EVIDENCE_active_count'] == 0
    assert q_row['EVIDENCE_active_precision'] is None and q_row['EVIDENCE_STOP_loss'] == 0.
    assert q_row['teacher_precision_among_q'] == 1.

    # SELF activity alone is not EVIDENCE activity; reaching pi stops every term.
    stopped = _fixture(15)
    for view in VIEWS:
        stopped['target_probabilities'][view][:] = PI
    stop_census = Census('U_s'); stop_census.add(**stopped)
    stop_row = stop_census.finish(15)['summary']
    assert all(stop_row[view + '_active_count'] == 0 for view in VIEWS)
    assert stop_row['SELF_STOP_loss'] == 0. and stop_row['EVIDENCE_STOP_loss'] == 0.
    self_only = _fixture(15); self_only['target_probabilities']['SELF'][:] = .5
    only_census = Census('U_s'); only_census.add(**self_only)
    only_row = only_census.finish(15)['summary']
    assert only_row['SELF_active_count'] == 15 and only_row['EVIDENCE_active_count'] == 0

    # A single accepted packet uses the complete B denominator, also across uneven batches.
    full = _fixture(100); full['accept_mask'][:] = False; full['accept_mask'][0] = True
    full['targets'][:] = torch.arange(100) % 6
    for view in VIEWS:
        full['target_probabilities'][view][:] = .5
    full_before = copy.deepcopy(full)
    full_census = Census('U_s'); full_census.add(**full)
    hidden = full_census.finish(100)
    hidden_row = hidden['summary']; expected = COEFFICIENT * math.log(PI / .5) / 100
    _close(hidden_row['SELF_STOP_loss'], expected); _close(hidden_row['EVIDENCE_STOP_loss'], expected)
    assert hidden_row['full_denominator_count'] == 100 and hidden_row['view_denominator_count'] == 200
    assert hidden_row['accepted_count'] == 1 and hidden_row['EVIDENCE_active_count'] == 1
    assert hidden_row['new_active_vs_SELF_count'] == 0 and hidden['truth_status'] == 'HIDDEN'
    for value in [hidden_row, *hidden['by_RX'].values(), *hidden['strata']]:
        assert value['teacher_precision_among_q'] is None and value['EVIDENCE_active_precision'] is None
        assert value['accepted_correct_count'] is None and value['accepted_error_count'] is None
        assert value['EVIDENCE_active_correct_count'] is None and value['EVIDENCE_active_error_count'] is None
    assert sum(hidden_row['predicted_class_counts'].values()) == 100
    assert _equal(full_before, full)
    uneven = Census('U_s')
    for left, right in ((0, 10), (10, 100)):
        part = {key: value[left:right] if key != 'target_probabilities' else
                {view: probabilities[left:right] for view, probabilities in value.items()}
                for key, value in full.items()}
        uneven.add(**part)
    uneven_result = uneven.finish(100)
    _close(uneven_result['summary']['EVIDENCE_STOP_loss'], expected)
    assert uneven_result['summary']['EVIDENCE_STOP_batch_loss_sum'] > expected * 9
    assert len(uneven_result['batches']) == 2
    assert uneven_result['summary']['target_probability_stats']['SELF']['all']['count'] == 100
    _close(uneven_result['summary']['target_probability_stats']['SELF']['all']['mean'], .5)
    json.dumps(hidden, allow_nan=False); json.dumps(uneven_result, allow_nan=False)

    # All protocol/shape/frozen-input failures reject atomically.
    clean = _fixture(3)
    for key, value in (
        ('source_truth', torch.zeros(3, dtype=torch.long)),
        ('accept_mask', torch.ones(3)), ('rx', torch.tensor([2, 3, 4])),
        ('day', torch.tensor([4, 1, 1])), ('targets', torch.tensor([0, 1, 6])),
        ('targets', torch.zeros(3)), ('targets', torch.zeros(2, dtype=torch.long)),
        ('accept_mask', torch.ones((3, 1), dtype=torch.bool))):
        invalid = copy.deepcopy(clean); invalid[key] = value
        reject(Census('U_s'), **invalid)
    for bad in (float('nan'), float('inf'), 0., -.1, 1.01):
        invalid = copy.deepcopy(clean); invalid['target_probabilities']['PA_WEAK'][0] = bad
        reject(Census('U_s'), **invalid)
    invalid = copy.deepcopy(clean); invalid['target_probabilities']['SELF'] = torch.ones(3, requires_grad=True)
    reject(Census('U_s'), **invalid)
    invalid = copy.deepcopy(clean); invalid['target_probabilities']['SELF'] = torch.ones((3, 1))
    reject(Census('U_s'), **invalid)
    invalid = copy.deepcopy(clean); invalid['target_probabilities']['EXTRA'] = torch.ones(3)
    reject(Census('U_s'), **invalid)
    invalid = copy.deepcopy(clean); invalid['physical_ids'][1] = invalid['physical_ids'][0]
    reject(Census('U_s'), **invalid)
    invalid = copy.deepcopy(clean)
    invalid['physical_ids'][1] = (*invalid['physical_ids'][1][:4], invalid['physical_ids'][0][4])
    reject(Census('U_s'), **invalid)
    invalid = copy.deepcopy(clean); first = invalid['physical_ids'][0]
    invalid['physical_ids'][0] = (first[0], 3, first[2], first[3], first[4])
    reject(Census('U_s'), **invalid)
    invalid = copy.deepcopy(clean); first = invalid['physical_ids'][0]
    invalid['physical_ids'][0] = (first[0], first[1], 2, first[3], first[4])
    reject(Census('U_s'), **invalid)
    invalid = copy.deepcopy(clean); first = invalid['physical_ids'][0]
    invalid['physical_ids'][0] = (False, *first[1:])
    reject(Census('U_s'), **invalid)
    duplicate = Census('U_s'); duplicate.add(**clean); reject(duplicate, **clean)
    new_id_same_base = _fixture(3, 100)
    new_id_same_base['physical_ids'][0] = (1, 1, 1, 100, 0)
    reject(duplicate, **new_id_same_base)
    partial_truth = Census('V_proxy'); partial_truth.add(**dict(clean, source_truth=torch.zeros(3, dtype=torch.long)))
    reject(partial_truth, **_fixture(3, 100))
    for bad_truth in (torch.tensor([0, 1, 6]), torch.zeros(2, dtype=torch.long),
                      torch.zeros(3, requires_grad=True)):
        reject(Census('V_proxy'), **dict(clean, source_truth=bad_truth))
    no_truth = Census('V_proxy'); no_truth.add(**clean)
    assert no_truth.finish(3)['truth_status'] == 'NOT_PROVIDED'
    assert no_truth.finish(3)['summary']['EVIDENCE_active_precision'] is None
    for bad_count in (2, 4, True, 0):
        try:
            duplicate.finish(bad_count)
        except ValueError:
            negative_count += 1
        else:
            raise AssertionError('Wrong full census denominator accepted')
    try:
        Census('target')
    except ValueError:
        negative_count += 1
    else:
        raise AssertionError('Forbidden role accepted')
    assert _equal(rng, _rng()), 'Frozen census changed Python/NumPy/Torch RNG'
    return dict(status='PASS', negative_checks=negative_count,
                overall_teacher_precision_counterexample=.99,
                EVIDENCE_active_precision_counterexample=0.,
                complete_batch_denominator=True, global_uneven_batch_denominator=True,
                hidden_U_truth=True, input_state_RNG_unchanged=True,
                source_RX_day_strata=15, data_reads=0, checkpoint_reads=0,
                model_or_optimizer_updates=0)


if __name__ == '__main__':
    print(json.dumps(check(), allow_nan=False))
