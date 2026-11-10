import pytest
from experiments.pamf_physics_branch.plan import RUN, SEED, candidate_rows, validate_candidate
from experiments.pamf_physics_branch.source import validate_source


def test_physics_pilot_has_matched_direction_controls():
    rows = candidate_rows()
    assert len(rows) == 2 and {r['beta'] for r in rows} == {0., .5}
    a, b = rows
    assert {k: v for k, v in a.items() if k not in ('row_id', 'beta')} == {
           k: v for k, v in b.items() if k not in ('row_id', 'beta')}
    assert a['source_rxs'] == [1, 3, 4, 6, 8] and a['echo_magnitude'] == [.04, .1]


def test_physics_pilot_rejects_target_or_unregistered_strength():
    row = candidate_rows()[0]
    with pytest.raises(ValueError):
        validate_candidate(dict(row, beta=.9))
    with pytest.raises(ValueError):
        validate_candidate(dict(row, target_truth='/forbidden'))


def test_source_guard_rejects_truth_and_wrong_pl_dose():
    row = candidate_rows()[0]
    c = dict(physical_row=row, run_id=RUN, row_id=row['row_id'], model_seed=SEED,
             source_rxs=row['source_rxs'], source_days=row['source_days'], num_classes=6,
             epochs=50, optimizer_updates=11100, ablation='F21', U_use=True, own_EMA=False,
             native_U_gate=False, posthoc_head_fit=False,
             paper_row=dict(structure='full', permuted=False, signature_views=3),
             hard_pl=dict(method='HPL2', threshold=.95, weight=1.), hard_pl_label_smoothing=0.,
             physical_relation=dict(beta=0., l_weight=.05, u_weight=.05))
    assert validate_source(c) == row
    with pytest.raises(ValueError, match='target'):
        validate_source(dict(c, truth='/forbidden'))
    with pytest.raises(ValueError, match='matched architecture'):
        validate_source(dict(c, hard_pl=dict(method='HPL2', threshold=.95, weight=2.)))
