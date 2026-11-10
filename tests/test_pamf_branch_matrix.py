import pytest

from experiments.pamf_branch_training.plan import candidate_rows, validate_candidate
from experiments.pamf_branch_training.plan import RUN, SEED
from experiments.pamf_branch_training.source import validate_source


def test_one_seed_matrix_separates_labeled_and_unlabeled_branch_losses():
    rows = candidate_rows()
    assert len(rows) == 4
    assert {r['model_seed'] for r in rows} == {2026092701}
    assert {(r['regime'], r['l_weight'], r['u_weight']) for r in rows} == {
        ('sup', .05, 0.), ('pl', .05, 0.), ('pl', 0., .05), ('pl', .05, .05)}
    assert all(r['structure'] == 'full' and not r['permuted'] for r in rows)


def test_reject_hidden_target_handles_and_wrong_parent_role():
    row = candidate_rows()[0]
    bad = dict(row, target_truth='/forbidden')
    with pytest.raises(ValueError):
        validate_candidate(bad)
    bad = dict(row, parent_regime='pl')
    with pytest.raises(ValueError):
        validate_candidate(bad)


def test_nonregistered_weight_is_rejected():
    with pytest.raises(ValueError):
        validate_candidate(dict(candidate_rows()[1], u_weight=.7))


def test_source_boundary_rejects_target_and_wrong_u_role():
    row = candidate_rows()[0]
    c = dict(branch_row=row, run_id=RUN, row_id=row['row_id'], model_seed=SEED,
             source_rxs=row['source_rxs'], source_days=row['source_days'], num_classes=6,
             epochs=50, optimizer_updates=11100, branch_relation=dict(l_weight=.05, u_weight=0.),
             ablation='F20', U_use=False, own_EMA=False, native_U_gate=False,
             posthoc_head_fit=False, paper_row=dict(structure='full', permuted=False, signature_views=3))
    assert validate_source(c) == row
    with pytest.raises(ValueError, match='target'):
        validate_source(dict(c, truth='/forbidden'))
    with pytest.raises(ValueError, match='contract'):
        validate_source(dict(c, U_use=True))
