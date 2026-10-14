"""Fixed, disjoint current/held banks from the original visible source L."""
from collections.abc import Mapping
from numbers import Integral

import torch


SOURCE_RX = (1, 3, 4, 6, 8)
SOURCE_DAYS = (1, 2, 3)
CLASSES = tuple(range(6))
CURRENT_PAIRS = ((0, 1), (17, 18), (35, 36), (52, 53))
HELD_PAIR = (68, 69)
L_COUNT = 6300
CELL_COUNT = 90
PACKETS_PER_CELL = 70
_INTEGER_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
_ID_FIELDS = ('eq_i', 'rx_i', 'day_i', 'sig_i', 'base_index')


def _integer(value, name):
    if isinstance(value, torch.Tensor):
        if value.requires_grad or value.ndim != 0 or value.dtype not in _INTEGER_DTYPES:
            raise ValueError(f'{name} requires an integer scalar')
        return int(value)
    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)
    raise ValueError(f'{name} requires an integer scalar, without truncation')


def _field(record, name):
    try:
        return _integer(record[name] if isinstance(record, Mapping) else getattr(record, name), name)
    except (KeyError, AttributeError) as error:
        raise ValueError(f'Original source metadata lacks {name}') from error


def _visible_source(dataset):
    # This precedes any access to index.tx_i, including for rejected hidden U.
    if getattr(dataset, 'hide', None) is not False or getattr(dataset, 'tx_label_visible', True) is not True:
        raise ValueError('Only the raw visible labeled source dataset is allowed')
    if getattr(dataset, 'split_source', None) != 'ssdg_labeled_tx_visible':
        raise ValueError('Original source L role is required; no U/V/target subset')
    if getattr(dataset, 'role', 'L_s') not in ('L_s', 'labeled_train'):
        raise ValueError('Non-L dataset role is forbidden')
    try:
        if len(dataset) != L_COUNT or len(dataset.index) != L_COUNT or len(dataset.selected) != L_COUNT:
            raise ValueError('Raw source L must contain exactly 6300 index/selected records')
    except (TypeError, AttributeError) as error:
        raise ValueError('Raw source L index and selected mapping are required') from error
    base = getattr(dataset, 'base', None)
    if (getattr(dataset, 'transform', None) is not None or getattr(base, 'transform', None) is not None
            or getattr(base, 'crop_mode', 'center') == 'random'):
        raise ValueError('Raw deterministic source IQ is required, without dataset augmentation')


def _actual_sample(dataset, index, record):
    sample = dataset[index]
    if not isinstance(sample, (tuple, list)) or len(sample) != 4:
        raise ValueError('Raw L sample must return IQ, label, domain and metadata')
    x, y, _, meta = sample
    if (not isinstance(x, torch.Tensor) or x.requires_grad or x.layout != torch.strided
            or x.shape != (2, 256) or x.dtype != torch.float32 or x.device.type != 'cpu'
            or not bool(torch.isfinite(x).all())):
        raise ValueError('Finite detached raw source IQ must be CPU float32[2,256]')
    if not isinstance(meta, Mapping):
        raise ValueError('Visible L sample physical metadata is required')
    if meta.get('tx_label_visible', True) is not True:
        raise ValueError('L sample hides TX labels')
    if meta.get('split_source', dataset.split_source) != dataset.split_source:
        raise ValueError('Sample source role differs from raw L dataset')
    if meta.get('meta_ssl_role', 'L_s') not in ('L_s', 'labeled_train') or 'true_tx_i' in meta:
        raise ValueError('Hidden or non-L sample metadata is forbidden')
    identity = tuple(_field(meta, field) for field in _ID_FIELDS)
    label = _integer(y, 'sample label')
    if identity != record['id'] or label != record['y'] or _field(meta, 'tx_i') != record['y']:
        raise ValueError('Actual IQ sample label/physical metadata differs from original L index')
    return x, label, identity


def banks(dataset):
    """Read metadata6300 and exactly900 distinct IQ; create no persistent role.

    Current keys are each ordinal pair's first ordinal: 0,17,35,52.
    Each bank has x/y/rx/day/side/ids, with side0=A and side1=B.
    Within each side, rows follow sorted(TX,RX,day); within-cell ordinal is
    the existing dataset.index order, never a new sort by sig/base_index.
    """
    _visible_source(dataset)
    records, cells = [], {}
    ids, base_indices = set(), set()
    for index, item in enumerate(dataset.index):
        y, rx, day = (_field(item, name) for name in ('tx_i', 'rx_i', 'day_i'))
        if y not in CLASSES or rx not in SOURCE_RX or day not in SOURCE_DAYS:
            raise ValueError('Source L TX/RX/day axes differ')
        eq, sig = _field(item, 'eq_i'), _field(item, 'sig_i')
        base_index = _integer(dataset.selected[index], 'selected base_index')
        identity = (eq, rx, day, sig, base_index)
        if any(value < 0 for value in identity) or identity in ids or base_index in base_indices:
            raise ValueError('Source L physical identities/base_index must be nonnegative and globally unique')
        ids.add(identity); base_indices.add(base_index)
        records.append(dict(y=y, rx=rx, day=day, id=identity))
        cells.setdefault((y, rx, day), []).append(index)
    expected_cells = {(y, rx, day) for y in CLASSES for rx in SOURCE_RX for day in SOURCE_DAYS}
    if set(cells) != expected_cells or any(len(indices) != PACKETS_PER_CELL for indices in cells.values()):
        raise ValueError('Original source L must have exactly90 cells x70 packets')
    keys = sorted(cells)
    groups = {first: [(side, cells[key][ordinal]) for side, ordinal in enumerate((first, second))
                      for key in keys] for first, second in CURRENT_PAIRS}
    groups['held'] = [(side, cells[key][ordinal]) for side, ordinal in enumerate(HELD_PAIR) for key in keys]
    selected = [index for group in groups.values() for _, index in group]
    selected_ids = [records[index]['id'] for index in selected]
    if (len(selected) != 900 or len(set(selected)) != 900 or len(set(selected_ids)) != 900
            or len({identity[4] for identity in selected_ids}) != 900):
        raise ValueError('All four current banks and held bank must be physically disjoint900')
    result = {}
    for name, group in groups.items():
        xs, ys, receivers, days, sides, physical_ids = [], [], [], [], [], []
        for side, index in group:
            record = records[index]
            x, y, identity = _actual_sample(dataset, index, record)
            xs.append(x); ys.append(y); receivers.append(record['rx']); days.append(record['day'])
            sides.append(side); physical_ids.append(identity)
        result[name] = dict(x=torch.stack(xs), y=torch.tensor(ys, dtype=torch.long),
                            rx=torch.tensor(receivers, dtype=torch.long), day=torch.tensor(days, dtype=torch.long),
                            side=torch.tensor(sides, dtype=torch.long), ids=physical_ids)
    return dict(current={first: result[first] for first, _ in CURRENT_PAIRS}, held=result['held'],
                summary=dict(metadata_count=L_COUNT, source_cell_count=CELL_COUNT,
                             packets_per_cell=PACKETS_PER_CELL, current_bank_count=4,
                             current_IQ_count=720, held_IQ_count=180, unique_IQ_count=900,
                             current_held_physical_disjoint=True, source_role='L_s',
                             source_RX=list(SOURCE_RX), source_days=list(SOURCE_DAYS),
                             current_ordinal_pairs=[list(pair) for pair in CURRENT_PAIRS],
                             held_ordinal_pair=list(HELD_PAIR), new_persistent_role=False))
