import numpy as np
import pytest
from experiments.orthogonal_paper_repro.training_data import select_training_records


def test_selects_only_designated_train_and_session_indices():
    payload = {'node_list': ['a', 'b', 'c'],
               'data': [np.arange(12).reshape(6, 2), np.arange(12, 24).reshape(6, 2),
                        np.arange(24, 36).reshape(6, 2)]}
    metadata = {'base': [{'id': 'a', 'indices': [0, 2, 4]}, {'id': 'b', 'indices': [1, 3, 5]}],
                'increments': [[{'id': 'c', 'indices': [0, 2, 4]}]]}
    staged = select_training_records(payload, metadata, validation_fraction=.34)
    assert set(staged) == {'base_train_x', 'base_train_y', 'base_validation_x',
                           'base_validation_y', 'session_1_x', 'session_1_y'}
    np.testing.assert_array_equal(staged['base_train_x'], payload['data'][0][[0, 2]].tolist() + payload['data'][1][[1, 3]].tolist())
    np.testing.assert_array_equal(staged['base_validation_x'], payload['data'][0][[4]].tolist() + payload['data'][1][[5]].tolist())
    np.testing.assert_array_equal(staged['session_1_x'], payload['data'][2][[0, 2, 4]])


def test_duplicate_and_out_of_range_indices_are_rejected():
    payload = {'node_list': ['a'], 'data': [np.zeros((4, 2))]}
    with pytest.raises(ValueError):
        select_training_records(payload, {'base': [{'id': 'a', 'indices': [0, 0, 5]}], 'increments': []})
