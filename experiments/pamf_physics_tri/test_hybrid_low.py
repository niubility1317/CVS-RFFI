from experiments.pamf_physics_tri import hybrid_low_plan as plan


def test_low_weight_matrix_and_same_setting_baselines():
    rows = plan.candidate_rows()
    assert len(rows) == len({row['row_id'] for row in rows}) == 24
    assert {row['auxiliary_weight'] for row in rows} == {.05, .10}
    assert len({row['setting'] for row in rows}) == 4
    assert all(plan.baseline_row(row)['arm'] == 'hybrid' for row in rows)
    assert all(plan.baseline_row(row)['model_seed'] == row['model_seed'] for row in rows)
