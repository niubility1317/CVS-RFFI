import pytest
from experiments.cvs_phase1_repair import evaluate_completed14 as e

def test_fixed_fourteen_rejects_reordered_missing_or_target_contaminated_freeze():
    rows=[dict(row_id=r) for r in e.ROW_IDS]
    value=dict(run_id=e.RUN,selection=e.SELECTION,rows=rows,target_access=False)
    assert e.validate_freeze(value,rows)==rows
    for changed in [dict(value,rows=rows[:-1]),dict(value,rows=list(reversed(rows))),dict(value,target_access=True),dict(value,selection='ranked_by_score')]:
        with pytest.raises(ValueError):e.validate_freeze(changed,rows)
