import pytest

from experiments.pamf_phase2_practical.clean_mopc_savedb_diagnostic import run


def test_rejects_unsupported_diagnostic_before_any_data_access():
    with pytest.raises(ValueError, match='diagnostic'):
        run({}, {}, 10, 20, 'cuda:0', 'unapproved', lambda x: None)

