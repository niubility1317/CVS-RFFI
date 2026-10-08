import json
from pathlib import Path
import pytest
from experiments.cvs_phase1_stack import evaluate_r5_completed12 as e

def test_twelve_fixed_completed_rows_and_freeze_rejects_changes():
    rows=json.loads((Path(e.__file__).parent/'r5_fixed_rows_20261008.json').read_text(encoding='utf-8'))
    value=dict(run_id=e.RUN,selection=e.SELECTION,rows=rows,target_access=False)
    assert len(rows)==12 and {c['arm'] for c in rows}=={'base','daot','both'}
    assert e.validate_freeze(value,rows)==rows
    for bad in [dict(value,rows=rows[:-1]),dict(value,target_access=True),dict(value,selection='score_ranked')]:
        with pytest.raises(ValueError):e.validate_freeze(bad,rows)
