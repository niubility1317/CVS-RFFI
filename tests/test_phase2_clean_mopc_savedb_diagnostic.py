import pytest

from experiments.pamf_phase2_practical.clean_mopc_savedb_diagnostic import run


def test_rejects_unsupported_diagnostic_before_any_data_access():
    with pytest.raises(ValueError, match='diagnostic'):
        run({}, {}, 10, 20, 'cuda:0', 'unapproved', lambda x: None)


def test_rejects_invalid_uninstrumented_repeat_count_before_data_access():
    with pytest.raises(ValueError, match='repeat'):
        run({}, {}, 10, 20, 'cuda:0', 'original_uninstrumented', lambda x: None, repeats=0)


def test_affine_candidate_still_rejects_unapproved_job():
    with pytest.raises(ValueError, match='diagnostic job'):
        run({}, {}, 10, 20, 'cuda:0', 'affine_bridge', lambda x: None)

