"""A fixed second, physically disjoint source-L subset for ROW confirmation.

Only the original raw visible L dataset is accepted. Permissions and actual
sample checks are the existing validated helpers; previous bank constants are
never changed. The returned IDs are private in-memory inputs to the assay,
while the public summary contains only selection/count facts.
"""
import torch

from experiments.cvs_identity_difference20.bank import (
    SOURCE_RX, SOURCE_DAYS, CLASSES, L_COUNT, CELL_COUNT, PACKETS_PER_CELL,
    _actual_sample, _field, _integer, _visible_source,
)

CURRENT_PAIRS = ((2, 3), (19, 20), (37, 38), (54, 55))
HELD_PAIR = (66, 67)
PREVIOUS_ORDINALS = (0, 1, 17, 18, 35, 36, 52, 53, 68, 69)


def _selection():
    if (not isinstance(CURRENT_PAIRS, tuple) or len(CURRENT_PAIRS) != 4
            or any(not isinstance(pair, tuple) or len(pair) != 2 for pair in CURRENT_PAIRS)
            or not isinstance(HELD_PAIR, tuple) or len(HELD_PAIR) != 2):
        raise ValueError('Four fixed current pairs and one held pair are required')
    ordinals = tuple(ordinal for pair in CURRENT_PAIRS + (HELD_PAIR,) for ordinal in pair)
    if (any(type(ordinal) is not int or not 0 <= ordinal < PACKETS_PER_CELL for ordinal in ordinals)
            or len(set(ordinals)) != 10 or set(ordinals).intersection(PREVIOUS_ORDINALS)):
        raise ValueError('New ordinals must be legal, mutually disjoint and disjoint from the previous ten ordinals')
    return ordinals


def banks(dataset):
    """Return old-style banks with current keys2/19/37/54 and held66/67.

    Inspect source metadata6300 and read exactly900 distinct raw source IQ.
    Each180-bank has side0/side1, each ordered sorted(TX,RX,day), with within
    cell positions taken from the unchanged original dataset.index order.
    Previous subset disjointness uses metadata only, not its IQ. This is a new
    current-step diagnostic subset of the original seen source L, not a new
    persistent split, unseen receiver set, or blind/generalization evidence.
    """
    _visible_source(dataset)  # Must precede any tx_i inspection.
    ordinals = _selection()
    records, cells = [], {}
    identities, base_indices = set(), set()
    for index, item in enumerate(dataset.index):
        y, rx, day = (_field(item, name) for name in ('tx_i', 'rx_i', 'day_i'))
        if y not in CLASSES or rx not in SOURCE_RX or day not in SOURCE_DAYS:
            raise ValueError('Source L TX/RX/day axes differ')
        eq, sig = _field(item, 'eq_i'), _field(item, 'sig_i')
        base_index = _integer(dataset.selected[index], 'selected base_index')
        identity = (eq, rx, day, sig, base_index)
        if any(value < 0 for value in identity) or identity in identities or base_index in base_indices:
            raise ValueError('Original source physical ID and base_index must be nonnegative and globally unique')
        identities.add(identity); base_indices.add(base_index)
        records.append(dict(y=y, rx=rx, day=day, id=identity))
        cells.setdefault((y, rx, day), []).append(index)
    expected = {(y, rx, day) for y in CLASSES for rx in SOURCE_RX for day in SOURCE_DAYS}
    if set(cells) != expected or any(len(indices) != PACKETS_PER_CELL for indices in cells.values()):
        raise ValueError('Original source L must be exactly90 cells x70 physical packets')
    keys = sorted(cells)
    groups = {first: [(side, cells[key][ordinal]) for side, ordinal in enumerate((first, second))
                      for key in keys] for first, second in CURRENT_PAIRS}
    groups['held'] = [(side, cells[key][ordinal]) for side, ordinal in enumerate(HELD_PAIR) for key in keys]
    selected = [index for group in groups.values() for _, index in group]
    selected_ids = [records[index]['id'] for index in selected]
    selected_bases = {identity[-1] for identity in selected_ids}
    previous = {cells[key][ordinal] for key in keys for ordinal in PREVIOUS_ORDINALS}
    previous_ids = {records[index]['id'] for index in previous}
    previous_bases = {identity[-1] for identity in previous_ids}
    if (len(selected) != 900 or len(set(selected)) != 900 or len(set(selected_ids)) != 900
            or len(selected_bases) != 900 or len(previous) != 900 or len(previous_ids) != 900
            or len(previous_bases) != 900 or set(selected).intersection(previous)
            or set(selected_ids).intersection(previous_ids) or selected_bases.intersection(previous_bases)):
        raise ValueError('New current/held900 and previous subset900 must have disjoint complete physical IDs/baseindices')
    result = {}
    for name, group in groups.items():
        xs, ys, receivers, days, sides, ids = [], [], [], [], [], []
        for side, index in group:
            record = records[index]
            x, y, identity = _actual_sample(dataset, index, record)
            xs.append(x); ys.append(y); receivers.append(record['rx']); days.append(record['day'])
            sides.append(side); ids.append(identity)
        result[name] = dict(x=torch.stack(xs), y=torch.tensor(ys, dtype=torch.long),
                            rx=torch.tensor(receivers, dtype=torch.long), day=torch.tensor(days, dtype=torch.long),
                            side=torch.tensor(sides, dtype=torch.long), ids=ids)
    return dict(current={first: result[first] for first, _ in CURRENT_PAIRS}, held=result['held'],
                summary=dict(metadata_count=L_COUNT, source_cell_count=CELL_COUNT,
                             packets_per_cell=PACKETS_PER_CELL, current_bank_count=4,
                             current_IQ_count=720, held_IQ_count=180, unique_IQ_count=900,
                             current_held_physical_disjoint=True, new_previous_physical_disjoint=True,
                             new_previous_baseindex_disjoint=True, new_previous_ordinal_disjoint=True,
                             previous_subset_metadata_count=len(previous), previous_subset_IQ_reads=0,
                             selected_new_ordinal_count=len(ordinals), source_role='L_s',
                             source_RX=list(SOURCE_RX), source_days=list(SOURCE_DAYS),
                             current_ordinal_pairs=[list(pair) for pair in CURRENT_PAIRS],
                             held_ordinal_pair=list(HELD_PAIR), previous_ordinals=list(PREVIOUS_ORDINALS),
                             new_persistent_role=False, already_seen_source_L=True))
