import pytest
from experiments.pamf_physics_branch.plan import RUN as OLD_RUN, candidate_rows as old_rows
from experiments.pamf_physics_branch.source import validate_source as old_validate
from experiments.pamf_physics_branch_balanced.plan import RUN, candidate_rows
from experiments.pamf_physics_branch_balanced.source import validate_source


def test_only_reference_weight_changes_from_matched_physical_pilot():
    old, new = old_rows()[1], candidate_rows()[0]
    assert {k: v for k, v in old.items() if k != 'row_id'} == {
        k: v for k, v in new.items() if k not in ('row_id', 'reference_scale')}
    assert new['reference_scale'] == .1


def test_balanced_source_guard_keeps_exact_contract_and_rejects_target():
    row = candidate_rows()[0]
    c = dict(physical_row=row, run_id=RUN, row_id=row['row_id'], model_seed=row['model_seed'],
             source_rxs=row['source_rxs'], source_days=row['source_days'], num_classes=6,
             epochs=50, optimizer_updates=11100, ablation='F21', U_use=True, own_EMA=False,
             native_U_gate=False, posthoc_head_fit=False,
             paper_row=dict(structure='full', permuted=False, signature_views=3),
             hard_pl=dict(method='HPL2', threshold=.95, weight=1.), hard_pl_label_smoothing=0.,
             physical_relation=dict(beta=.5, l_weight=.05, u_weight=.05), reference_scale=.1,
             algorithm_id='pamf_ptbr_refbudget_v1', output_root='/project/runs/' + RUN + '/F/' + row['row_id'] + '/source')
    assert validate_source(c) == row
    with pytest.raises(ValueError):
        validate_source(dict(c, reference_scale=.2))
    with pytest.raises(ValueError, match='target'):
        validate_source(dict(c, target_truth='/forbidden'))
    original = dict(c, run_id=OLD_RUN, row_id=old_rows()[1]['row_id'], physical_row=old_rows()[1])
    with pytest.raises(ValueError):
        old_validate(original)
