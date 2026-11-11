import copy
import json
from pathlib import Path

import pytest

from experiments.pamf_tri_phase_splits import plan


def test_split_matrix_and_forbidden_parent_or_truth():
    context_path = Path(r'E:\type10-7\local_artifacts\tri_unprotected_context.json')
    if not context_path.exists():
        pytest.skip('Local source-handle context unavailable')
    context = json.loads(context_path.read_text(encoding='utf-8'))
    rows = plan.candidate_rows()
    assert len(rows) == 12
    for row in rows:
        c = plan.configuration(context, row, Path('/test/project'), 'test')
        plan.validate_candidate(c)
        bad = copy.deepcopy(c)
        bad['shared_prefix'] = bad['shared_prefix'].replace(str(row['model_seed']), '2026092799')
        with pytest.raises(ValueError):
            plan.validate_candidate(bad)
        bad = copy.deepcopy(c)
        bad['target_truth'] = '/forbidden'
        with pytest.raises(ValueError):
            plan.validate_candidate(bad)
        bad = copy.deepcopy(c)
        bad['phase_subset'] = ['frequency', 'pa']
        with pytest.raises(ValueError):
            plan.validate_candidate(bad)
