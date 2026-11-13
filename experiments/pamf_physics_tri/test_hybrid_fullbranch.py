from experiments.pamf_physics_tri import hybrid_fullbranch_plan as plan


def test_fullbranch_control_matches_structural_c_matrix():
    rows=plan.candidate_rows()
    assert len(rows)==len({r['row_id'] for r in rows})==4
    assert {r['structure_arm'] for r in rows}=={'E'}
    assert {r['auxiliary_weight'] for r in rows}=={.1}
    assert {(r['setting'],r['model_seed']) for r in rows}=={
        ('main_split',2026092701),('ten_class',2026092701),('ten_class',2026092702),('ten_class',2026092703)}
