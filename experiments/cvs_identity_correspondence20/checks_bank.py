"""Focused new-subset checks using the existing synthetic source fixture."""
from collections import Counter
import copy
import json
from unittest.mock import patch

import torch

from . import bank as api
from experiments.cvs_identity_difference20.checks_bank import Dummy, READ_LOG, _equal, _rng


def _reject(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError('Invalid new source selection was accepted')


def check():
    READ_LOG.clear()
    dataset = Dummy()
    before = copy.deepcopy(dataset.__dict__)
    rng = _rng()
    result = api.banks(dataset)
    assert list(result['current']) == [2, 19, 37, 54]
    assert len(READ_LOG) == 900 and len(set(READ_LOG)) == 900
    assert result['summary']['unique_IQ_count'] == 900
    assert result['summary']['current_IQ_count'] == 720 and result['summary']['held_IQ_count'] == 180
    assert result['summary']['new_previous_physical_disjoint']
    assert result['summary']['new_previous_baseindex_disjoint']
    assert result['summary']['previous_subset_IQ_reads'] == 0
    assert not any(name in result['summary'] for name in ('ids', 'physical_ids', 'base_indices'))
    cells = {}
    for index, item in enumerate(dataset.index):
        cells.setdefault((item.tx_i, item.rx_i, item.day_i), []).append(index)
    old_locals = {indices[ordinal] for indices in cells.values() for ordinal in api.PREVIOUS_ORDINALS}
    assert len(old_locals) == 900 and not old_locals.intersection(READ_LOG)
    old_base = {int(dataset.selected[index]) for index in old_locals}
    groups = [(result['current'][pair[0]], pair) for pair in api.CURRENT_PAIRS]
    groups.append((result['held'], api.HELD_PAIR))
    selected_ids, expected_reads = [], []
    for bank, pair in groups:
        assert set(bank) == {'x', 'y', 'rx', 'day', 'side', 'ids'}
        assert bank['x'].shape == (180, 2, 256) and bank['x'].dtype == torch.float32
        assert bank['x'].device.type == 'cpu' and not bank['x'].requires_grad
        assert Counter(bank['side'].tolist()) == {0: 90, 1: 90}
        assert Counter(bank['rx'].tolist()) == {receiver: 36 for receiver in api.SOURCE_RX}
        assert Counter(bank['y'].tolist()) == {label: 30 for label in range(6)}
        exposures = Counter(zip(bank['side'].tolist(), bank['rx'].tolist(), bank['y'].tolist()))
        assert exposures == {(side, receiver, label): 3 for side in (0, 1)
                             for receiver in api.SOURCE_RX for label in range(6)}
        for side, ordinal in enumerate(pair):
            for position, key in enumerate(sorted(cells)):
                local = cells[key][ordinal]
                row = side * 90 + position
                expected_reads.append(local)
                item = dataset.index[local]
                expected_id = (item.eq_i, item.rx_i, item.day_i, item.sig_i, int(dataset.selected[local]))
                assert bank['ids'][row] == expected_id
                assert item.sig_i == 69 - ordinal
                assert float(bank['x'][row, 0, 0]) == expected_id[-1]
                assert (int(bank['y'][row]), int(bank['rx'][row]), int(bank['day'][row])) == key
                assert len(expected_id) == 5 and all(type(value) is int for value in expected_id)
        selected_ids.extend(bank['ids'])
    assert READ_LOG == expected_reads, 'Selection reordered the original within-cell index'
    assert len(set(selected_ids)) == 900 and len({identity[-1] for identity in selected_ids}) == 900
    assert not old_base.intersection(identity[-1] for identity in selected_ids)
    assert _equal(before, dataset.__dict__) and _equal(rng, _rng())
    negatives = 0
    # No metadata iteration/TX access is allowed before source visibility.
    class ForbiddenIndex:
        def __len__(self):
            return 6300

        def __iter__(self):
            raise AssertionError('TX metadata was read before source permission')

    for changed in ({'hide': True}, {'split_source': 'ssdg_unlabeled_tx_hidden'},
                    {'role': 'U_s'}, {'role': 'V'}, {'role': 'target'}):
        denied = Dummy()
        denied.index = ForbiddenIndex()
        for key, value in changed.items():
            setattr(denied, key, value)
        reads_before = len(READ_LOG)
        _reject(lambda: api.banks(denied))
        assert len(READ_LOG) == reads_before
        negatives += 1
    # A changed selection must fail before IQ reads if it overlaps either
    # previous ordinals or another new bank; old module constants stay intact.
    for name, value in (
        ('CURRENT_PAIRS', ((0, 3), (19, 20), (37, 38), (54, 55))),
        ('CURRENT_PAIRS', ((2, 3), (19, 20), (37, 38), (54, 54))),
        ('HELD_PAIR', (66, 2)), ('HELD_PAIR', (68, 67)),
        ('HELD_PAIR', (66, 70)), ('CURRENT_PAIRS', ((2, 3), (19, 20), (37, 38))),
    ):
        reads_before = len(READ_LOG)
        with patch.object(api, name, value):
            _reject(lambda: api.banks(dataset))
        assert len(READ_LOG) == reads_before
        negatives += 1
    from experiments.cvs_identity_difference20 import bank as previous
    assert previous.CURRENT_PAIRS == ((0, 1), (17, 18), (35, 36), (52, 53))
    assert previous.HELD_PAIR == (68, 69)
    assert _equal(before, dataset.__dict__) and _equal(rng, _rng())
    return dict(status='PASS', metadata_records=6300, current_keys=[2, 19, 37, 54],
                current_IQ=720, held_IQ=180, unique_new_IQ=900, previous_subset_IQ_reads=0,
                new_previous_physical_and_baseindex_disjoint=True,
                original_ordinal_order_and_side_RX_class_day_exposure=True,
                hidden_before_TX_and_IQ=True, negative_checks=negatives,
                input_dataset_previous_constants_and_RNG_unchanged=True,
                real_data_checkpoint_reads_and_updates=0)


if __name__ == '__main__':
    print(json.dumps(check(), allow_nan=False, ensure_ascii=False, indent=2))
