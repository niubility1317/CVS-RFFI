"""Dummy-only bank tests; no real IQ, model or checkpoint is accessed."""
from collections import Counter
import copy
from dataclasses import dataclass, replace
import json
import random
from types import SimpleNamespace

import numpy as np
import torch

from .bank import banks, CURRENT_PAIRS, HELD_PAIR, SOURCE_RX, SOURCE_DAYS


READ_LOG = []


@dataclass(frozen=True)
class Record:
    tx_i: int
    rx_i: int
    day_i: int
    eq_i: int
    sig_i: int


class Dummy:
    def __init__(self):
        self.hide = False
        self.tx_label_visible = True
        self.split_source = 'ssdg_labeled_tx_visible'
        self.transform = None
        self.base = SimpleNamespace(transform=None, crop_mode='center')
        self.index, self.original = [], {}
        keys = [(y, receiver, day) for y in range(6) for receiver in SOURCE_RX for day in SOURCE_DAYS]
        for ordinal in range(70):
            # Input is interleaved by ordinal, with rotated cell order and
            # descending sig_i; sorting by sig_i/base would choose wrong IQ.
            rotated = keys[ordinal % 90:] + keys[:ordinal % 90]
            for y, receiver, day in rotated:
                record = Record(y, receiver, day, 1, 69 - ordinal)
                local = len(self.index)
                self.index.append(record); self.original[10000 + local] = record
        self.selected = np.arange(10000, 16300, dtype=np.int64)
        self.sample_overrides = {}
        self.length = 6300

    def __len__(self):
        return self.length

    def __getitem__(self, local):
        READ_LOG.append(local)
        base = int(self.selected[local]); item = self.original[base]
        x = torch.full((2, 256), float(base), dtype=torch.float32)
        label = item.tx_i
        metadata = {key: getattr(item, key) for key in ('eq_i', 'rx_i', 'day_i', 'sig_i', 'tx_i')}
        metadata.update(base_index=base, split_source=self.split_source, tx_label_visible=True)
        change = self.sample_overrides.get(local, {})
        metadata.update(change.get('meta', {}))
        return change.get('x', x), change.get('y', label), 0, metadata


def _equal(left, right):
    if isinstance(left, torch.Tensor):
        if (not isinstance(right, torch.Tensor) or left.shape != right.shape or left.dtype != right.dtype
                or left.requires_grad != right.requires_grad):
            return False
        return torch.equal(left, right) or (left.is_floating_point()
            and torch.equal(torch.isnan(left), torch.isnan(right))
            and torch.equal(torch.nan_to_num(left), torch.nan_to_num(right)))
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and set(left) == set(right) and all(_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, SimpleNamespace):
        return isinstance(right, SimpleNamespace) and _equal(vars(left), vars(right))
    return left == right


def _rng():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def check():
    rng = _rng()
    READ_LOG.clear()
    dataset = Dummy(); before = copy.deepcopy(dataset.__dict__)
    observed = banks(dataset)
    assert _equal(before, dataset.__dict__), 'Bank changed dataset/index/selected/sample state'
    assert list(observed['current']) == [0, 17, 35, 52]
    assert observed['summary']['metadata_count'] == 6300 and observed['summary']['unique_IQ_count'] == 900
    assert observed['summary']['current_IQ_count'] == 720 and observed['summary']['held_IQ_count'] == 180
    assert observed['summary']['new_persistent_role'] is False
    assert len(READ_LOG) == 900 and len(set(READ_LOG)) == 900
    cells = {}
    for local, item in enumerate(dataset.index):
        cells.setdefault((item.tx_i, item.rx_i, item.day_i), []).append(local)
    assert len(cells) == 90 and all(len(indices) == 70 for indices in cells.values())
    all_ids = []
    groups = [(observed['current'][pair[0]], pair) for pair in CURRENT_PAIRS]
    groups.append((observed['held'], HELD_PAIR))
    for bank, pair in groups:
        assert set(bank) == {'x', 'y', 'rx', 'day', 'side', 'ids'}
        assert bank['x'].shape == (180, 2, 256) and bank['x'].dtype == torch.float32
        assert all(bank[key].shape == (180,) and bank[key].dtype == torch.long for key in ('y', 'rx', 'day', 'side'))
        assert not any(bank[key].requires_grad for key in ('x', 'y', 'rx', 'day', 'side'))
        assert Counter(bank['side'].tolist()) == {0: 90, 1: 90}
        assert Counter(bank['rx'].tolist()) == {receiver: 36 for receiver in SOURCE_RX}
        assert Counter(bank['y'].tolist()) == {y: 30 for y in range(6)}
        triples = Counter(zip(bank['side'].tolist(), bank['rx'].tolist(), bank['y'].tolist()))
        assert triples == {(side, receiver, y): 3 for side in (0, 1) for receiver in SOURCE_RX for y in range(6)}
        for side, ordinal in enumerate(pair):
            for key_index, key in enumerate(sorted(cells)):
                row = side * 90 + key_index; local = cells[key][ordinal]
                item = dataset.index[local]
                expected_id = (item.eq_i, item.rx_i, item.day_i, item.sig_i, int(dataset.selected[local]))
                assert bank['ids'][row] == expected_id
                assert float(bank['x'][row, 0, 0]) == expected_id[-1]
                assert item.sig_i == 69 - ordinal, 'Original within-cell order was changed'
                assert (int(bank['y'][row]), int(bank['rx'][row]), int(bank['day'][row]), int(bank['side'][row])) == (*key, side)
        assert len(set(bank['ids'])) == 180 and len({identity[4] for identity in bank['ids']}) == 180
        assert all(len(identity) == 5 and all(type(value) is int for value in identity) for identity in bank['ids'])
        all_ids.extend(bank['ids'])
    assert len(all_ids) == 900 and len(set(all_ids)) == 900 and len({identity[-1] for identity in all_ids}) == 900
    held_ids = set(observed['held']['ids'])
    assert all(not held_ids & set(bank['ids']) for bank in observed['current'].values())

    negative_count = 0
    def reject(changes, *, no_IQ=False):
        nonlocal negative_count
        bad = Dummy(); changes(bad)
        saved = copy.deepcopy(bad.__dict__); reads_before = len(READ_LOG)
        try:
            banks(bad)
        except ValueError:
            negative_count += 1
        else:
            raise AssertionError('Invalid source bank input was accepted')
        assert _equal(saved, bad.__dict__), 'Rejected bank changed input dataset'
        if no_IQ:
            assert len(READ_LOG) == reads_before, 'Invalid source role read IQ before rejection'

    reject(lambda ds: setattr(ds, 'hide', True), no_IQ=True)
    reject(lambda ds: setattr(ds, 'tx_label_visible', False), no_IQ=True)
    reject(lambda ds: setattr(ds, 'split_source', 'ssdg_unlabeled_tx_hidden'), no_IQ=True)
    reject(lambda ds: setattr(ds, 'split_source', 'ssdg_source_v_select'), no_IQ=True)
    reject(lambda ds: setattr(ds, 'role', 'U_s'), no_IQ=True)
    reject(lambda ds: setattr(ds, 'length', 6299), no_IQ=True)
    reject(lambda ds: setattr(ds, 'index', ds.index[:-1]), no_IQ=True)
    reject(lambda ds: setattr(ds, 'selected', ds.selected[:-1]), no_IQ=True)
    reject(lambda ds: setattr(ds, 'transform', lambda x: x), no_IQ=True)
    reject(lambda ds: setattr(ds.base, 'crop_mode', 'random'), no_IQ=True)
    reject(lambda ds: ds.selected.__setitem__(1, ds.selected[0]), no_IQ=True)
    reject(lambda ds: ds.selected.__setitem__(1, -1), no_IQ=True)
    reject(lambda ds: ds.index.__setitem__(0, replace(ds.index[0], rx_i=2)), no_IQ=True)
    reject(lambda ds: ds.index.__setitem__(0, replace(ds.index[0], day_i=0)), no_IQ=True)
    reject(lambda ds: ds.index.__setitem__(0, replace(ds.index[0], tx_i=6)), no_IQ=True)
    reject(lambda ds: ds.index.__setitem__(0, replace(ds.index[0], tx_i=.5)), no_IQ=True)
    reject(lambda ds: ds.index.__setitem__(0, replace(ds.index[0], eq_i=-1)), no_IQ=True)
    reject(lambda ds: ds.index.__setitem__(0, replace(ds.index[0], sig_i=-1)), no_IQ=True)
    # Valid axes but 69/71 cells is still not the original balanced70 contract.
    reject(lambda ds: ds.index.__setitem__(0, replace(ds.index[0], tx_i=1)), no_IQ=True)
    for name, value in (('rx_i', 3), ('day_i', 2), ('tx_i', 1), ('eq_i', 0),
                        ('sig_i', 2), ('base_index', 0), ('tx_i', .5),
                        ('tx_label_visible', False), ('split_source', 'ssdg_source_v_select'),
                        ('meta_ssl_role', 'unlabeled_source'), ('true_tx_i', 0)):
        reject(lambda ds, name=name, value=value: ds.sample_overrides.__setitem__(0, {'meta': {name: value}}))
    reject(lambda ds: ds.sample_overrides.__setitem__(0, {'y': -1}))
    reject(lambda ds: ds.sample_overrides.__setitem__(0, {'y': 1}))
    reject(lambda ds: ds.sample_overrides.__setitem__(0, {'x': torch.zeros(2, 255)}))
    reject(lambda ds: ds.sample_overrides.__setitem__(0, {'x': torch.zeros(2, 256, dtype=torch.float64)}))
    reject(lambda ds: ds.sample_overrides.__setitem__(0, {'x': torch.full((2, 256), float('nan'))}))
    reject(lambda ds: ds.sample_overrides.__setitem__(0, {'x': torch.zeros(2, 256, requires_grad=True)}))
    # Proof that hidden U is rejected before tx_i is even inspected.
    class ForbiddenIndex:
        def __len__(self): return 6300
        def __iter__(self): raise AssertionError('Hidden U TX metadata was inspected')
    hidden = Dummy(); hidden.hide = True; hidden.index = ForbiddenIndex()
    try:
        banks(hidden)
    except ValueError:
        negative_count += 1
    else:
        raise AssertionError('Hidden U metadata was consumed')
    assert _equal(rng, _rng()), 'Bank changed Python/NumPy/Torch RNG'
    return dict(status='PASS', metadata_records=6300, selected_unique_IQ=900,
                current_banks=4, current_IQ=720, independent_held_IQ=180,
                original_70_packet_cell_order=True, side_RX_class_three_day_balance=True,
                full_physical_and_baseindex_disjoint=True, input_dataset_RNG_unchanged=True,
                hidden_U_rejected_before_TX_metadata=True, negative_checks=negative_count,
                real_data_reads=0, checkpoint_reads=0, actual_updates=0)


if __name__ == '__main__':
    print(json.dumps(check(), allow_nan=False))
