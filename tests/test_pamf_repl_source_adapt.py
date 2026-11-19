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


def test_noise_aware_screen_requires_gain_both_folds_and_limits_each_cell_loss():
    folds = {'fold1': ('a', 'b'), 'fold2': ('c', 'd')}
    scores = {(f, g): (1800, 24, 3) for f, groups in folds.items() for g in groups}
    scores['fold1', 'a'] = (1800, 0, 4)
    harmful = scores.copy()
    harmful['fold1', 'a'] = (1800, 0, 5)
    evidence = {4.5: scores, 3.: harmful}
    assert choose_weight(folds, evidence)['weight'] == 6.
    decision = choose_weight(folds, evidence, noise_aware=True)
    assert decision['weight'] == 4.5
    assert decision['candidates'][4.5]['paired_gain_pvalue'] <= .025
    assert not decision['candidates'][3.]['eligible']
    borderline = {(f, g): (1800, 2, 1) for f, groups in folds.items() for g in groups}
    assert choose_weight(folds, {4.5: borderline, 3.: borderline}, noise_aware=True)['weight'] == 6.


def test_transfer_applies_the_same_frozen_dose_to_each_seed_and_reuses_fallback():
    from experiments.pamf_repl_source_adapt import transfer

    native = dict(weight=6., decision='retain_native')
    accepted = dict(weight=4.5, decision='source_improvement', candidates={'4.5': {'eligible': True}})
    decisions = {setting: native for setting in transfer.SETTINGS}
    setting = transfer.SETTINGS[0]
    decisions[setting] = accepted
    payload = dict(status='SOURCE_ADAPTATION_DECISIONS_FROZEN', selected_seed=transfer.SEED,
                   target_access=False, U_truth_read=False, policy='development_calibration', decisions=decisions)
    rows = transfer.selected_rows(payload)
    changed = [row for row in rows if not row['reuses_original']]
    assert {row['model_seed'] for row in changed} == set(transfer.paper.SEEDS)
    assert len(rows) == 12 and len(changed) == 3
    assert all(row['setting'] == setting and row['weight'] == 4.5 for row in changed)
    with pytest.raises(ValueError, match='source decisions'):
        transfer.selected_rows(dict(payload, target_access=True))


def test_calibration_decisions_work_before_and_after_json_round_trip(tmp_path, monkeypatch):
    import json
    from experiments.cric_dura_comparison.io import write
    from experiments.pamf_repl_source_adapt import transfer

    write(tmp_path/'pipeline_state.json', {'status': 'ANALYZED'})
    write(tmp_path/'analysis'/'source_decisions.json', dict(selected_seed=transfer.SEED, target_access=False, U_truth_read=False))
    folds = {'f1': ('g1',), 'f2': ('g2',)}
    positive = {('f1', 'g1'): (1800, 30, 2), ('f2', 'g2'): (1800, 30, 2)}
    zero = {key: (1800, 0, 0) for key in positive}
    monkeypatch.setattr(transfer, 'collect_evidence', lambda *_: (
        dict.fromkeys(transfer.SETTINGS, folds), dict.fromkeys(transfer.SETTINGS, {4.5: positive, 3.: zero})))
    payload = transfer.calibration_decisions(tmp_path)
    assert transfer.selected_rows(payload) == transfer.selected_rows(json.loads(json.dumps(payload)))
    assert all(row['weight'] == 4.5 for row in transfer.selected_rows(payload))


def test_receiver_balance_keeps_existing_pair_support_and_avoids_single_partner_jitter():
    import torch
    from experiments.pamf_repl_source_adapt.receiver_balance import positive_target

    labels = torch.tensor([0, 0, 0, 0, 0, 1])
    receivers = torch.tensor([1, 2, 2, 2, 3, 4])
    target, stats = positive_target(labels, receivers)
    torch.testing.assert_close(target[0], torch.tensor([0., 5/24, 5/24, 5/24, 3/8, 0.]))
    assert target[5].sum() == 0 and stats['affected'] > 0 and stats['tv'] > 0
    collapsed = receivers.clamp_max(2)
    native, stats = positive_target(labels, collapsed)
    assert native is None and stats['affected'] == 0 and stats['tv'] == 0


def test_receiver_balance_confirmation_fixes_joint_route_and_excludes_development_seed():
    from experiments.pamf_repl_source_adapt import receiver_balance as balance

    rows = balance.rows(balance.CONFIRM_RUN)
    assert {row['route'] for row in rows} == {'joint'}
    assert {row['model_seed'] for row in rows} == {2026092701, 2026092703}
    assert {row['setting'] for row in rows} == set(balance.SETTINGS)
    assert len({row['row_id'] for row in rows}) == 8


def test_receiver_balance_dose_matrix_holds_fusion_and_the_compared_seed_fixed():
    from experiments.pamf_repl_source_adapt import receiver_balance as balance

    rows = balance.rows(balance.DOSE_RUN)
    assert len({row['row_id'] for row in rows}) == 12
    assert {row['model_seed'] for row in rows} == {balance.SEED}
    assert {row['route'] for row in rows} == {'joint'}
    for setting in balance.SETTINGS:
        assert {row['weight'] for row in rows if row['setting'] == setting} == {5.25, 4.5, 3.}
    verification = balance.rows(balance.DOSE_CONFIRM_RUN)
    assert {row['model_seed'] for row in verification} == {2026092701, 2026092703}
    assert len(verification) == 2 and all(row['setting'] == 'receiver_split_2' and row['weight'] == 3. for row in verification)
