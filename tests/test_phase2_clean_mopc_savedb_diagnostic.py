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


def test_original_diagnostic_rejects_saved_affine_job_before_data_access():
    job = dict(profile='protonet_cda', methods=['mopc_hr'], save_B_for_mopc=True,
               points=[[10, 20]], mopc_bridge='proto_affine_v1')
    with pytest.raises(ValueError, match='original bridge'):
        run({}, job, 10, 20, 'cuda:0', 'original_uninstrumented', lambda x: None)


def test_affine_diagnostic_rejects_dadda_before_data_access():
    job = dict(profile='dadda_sda', methods=['mopc_hr'], save_B_for_mopc=True, points=[[1, 20]])
    with pytest.raises(ValueError, match='ProtoNet'):
        run({}, job, 1, 20, 'cuda:0', 'affine_bridge', lambda x: None)

