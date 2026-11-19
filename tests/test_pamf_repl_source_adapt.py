import numpy as np
import pytest

from experiments.pamf_repl_source_adapt.controller import choose_weight
from experiments.pamf_repl_source_adapt.folds import ROLES, fold_groups, fold_positions


def _source_metadata():
    rx = np.repeat((10, 20, 30), 144)
    day = np.tile(np.repeat((1, 2, 3), 48), 3)
    common = dict(rx_i=rx, day_i=day)
    return {role: dict(common, y=np.arange(len(rx)) % 6) if role != 'U_s' else dict(common) for role in ROLES}


def test_receiver_fold_excludes_all_held_rx_from_actual_roles():
    metadata = _source_metadata()
    folds = fold_groups((30, 20, 10), (3, 2, 1))
    out = fold_positions(metadata, folds['rx_10'])
    for role, kept in out['train'].items():
        assert 10 not in metadata[role]['rx_i'][kept]
    assert sum(map(len, out['held_v'].values())) == 144
    metadata['U_s']['y'] = metadata['L_s']['y']
    with pytest.raises(ValueError, match='U truth'):
        fold_positions(metadata, folds['rx_10'])


def test_two_rx_folds_keep_every_rx_and_day_and_single_rx_defaults():
    assert fold_groups((10,), (1, 2, 3)) == {}
    pool = {(rx, date) for rx in (10, 20) for date in (1, 2, 3)}
    folds = fold_groups((10, 20), (1, 2, 3))
    assert len(folds) == 2
    assert not set(folds['cells_0']) & set(folds['cells_1'])
    for held in folds.values():
        remaining = pool - set(held)
        assert {x for x, _ in remaining} == {10, 20}
        assert {d for _, d in remaining} == {1, 2, 3}


def test_selection_rejects_a_hurt_group_even_when_pooled_gain_is_larger():
    folds = {'a': ('first', 'second'), 'b': ('third',)}
    def evidence(first, second, third):
        return {('a', 'first'): (100, first, 4), ('a', 'second'): (100, second, 4), ('b', 'third'): (80, third, 4)}

    stronger_but_uneven = evidence(3, 30, 30)
    modest = evidence(4, 6, 5)
    chosen = choose_weight(folds, {4.5: modest, 3.: stronger_but_uneven})
    assert chosen['weight'] == 4.5 and not chosen['candidates'][3.]['eligible']
    assert choose_weight(folds, {4.5: stronger_but_uneven, 3.: stronger_but_uneven})['weight'] == 6.
    assert choose_weight(folds, {4.5: modest, 3.: modest})['weight'] == 4.5
    assert choose_weight({}, {})['decision'] == 'insufficient_source_folds'


def test_rejects_partial_evidence_instead_of_treating_missing_arm_as_default():
    with pytest.raises(ValueError, match='complete'):
        choose_weight({'a': ('g1',), 'b': ('g2',)}, {4.5: {}})
