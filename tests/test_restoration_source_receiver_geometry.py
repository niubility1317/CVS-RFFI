import json

import numpy as np
import pytest

from experiments.pamf_phase3_restoration.source_receiver_geometry import (
    RUN,
    load_completed_reference,
    summarize_geometry,
)


def test_reference_variance_decomposition_and_zero_paired_error():
    reference = np.array([[0.], [2.], [4.], [6.]])
    q = summarize_geometry(reference, reference, np.array([0, 0, 1, 1]))
    assert q['total_variance'] == 5.
    assert q['between_group_variance'] == 4.
    assert q['within_group_variance'] == 1.
    assert q['clean_group_pattern_alignment_gain'] == 1.
    assert q['paired_feature_mse'] == q['paired_error_between_group'] == 0.


def test_group_pattern_amplification_is_distinct_from_global_bias():
    reference = np.array([[0.], [2.], [4.], [6.]])
    q = summarize_geometry(2*reference, reference, np.array([0, 0, 1, 1]))
    assert q['clean_group_pattern_alignment_gain'] == 2.
    assert q['paired_feature_mse'] == 14.
    assert q['paired_error_global_bias'] == 9.
    assert q['paired_error_between_group'] == 4.
    assert q['paired_error_within_group'] == 1.
    assert q['clean_group_pattern_error'] == 4.


def test_common_shift_does_not_look_like_receiver_pattern_amplification():
    reference = np.array([[0.], [2.], [4.], [6.]])
    q = summarize_geometry(reference+5, reference, np.array([0, 0, 1, 1]))
    assert q['paired_feature_mse'] == q['paired_error_global_bias'] == 25.
    assert q['paired_error_between_group'] == q['clean_group_pattern_error'] == 0.
    assert q['clean_group_pattern_alignment_gain'] == 1.


def test_incomplete_source_predictions_are_rejected_before_feature_access(tmp_path):
    folder = tmp_path/'runs'/RUN
    folder.mkdir(parents=True)
    (folder/'complete.json').write_text(json.dumps({'status': 'PENDING'}))
    with pytest.raises(ValueError, match='Completed frozen source V'):
        load_completed_reference(tmp_path)


def test_wrong_producer_is_rejected_before_feature_access(tmp_path):
    folder = tmp_path/'runs'/RUN
    folder.mkdir(parents=True)
    (folder/'complete.json').write_text(json.dumps({'status': 'ARTIFACTS_COMPLETE'}))
    (folder/'predictions_complete.json').write_text(json.dumps({
        'status': 'SOURCE_V_PREDICTIONS_COMPLETE', 'code_commit': 'wrong',
    }))
    with pytest.raises(ValueError, match='Exact frozen source V producer'):
        load_completed_reference(tmp_path)
