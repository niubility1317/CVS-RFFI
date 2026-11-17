import pickle
import numpy as np
import pytest
from experiments.orthogonal_paper_repro.query_pack import select_query_records, load_source


def test_query_pack_keeps_train_indices_disjoint_and_truth_separate():
    payload = {'node_list': ['a', 'b'], 'data': [np.arange(12).reshape(6, 2), np.arange(12, 24).reshape(6, 2)]}
    train = {'base': [{'id': 'a', 'indices': [0, 1]}], 'increments': [[{'id': 'b', 'indices': [0, 1]}]]}
    truth = {'classes': {'a': [2, 3], 'b': [4, 5]}}
    inputs, offsets, labels = select_query_records(payload, train, truth)
    np.testing.assert_array_equal(inputs, [[4, 5], [6, 7], [20, 21], [22, 23]])
    assert offsets == [2, 4]
    np.testing.assert_array_equal(labels[0], [0, 0])
    np.testing.assert_array_equal(labels[1], [0, 0, 1, 1])


def test_rejects_overlap_with_training_records():
    payload = {'node_list': ['a'], 'data': [np.zeros((4, 2))]}
    train = {'base': [{'id': 'a', 'indices': [0, 1]}], 'increments': []}
    with pytest.raises(ValueError):
        select_query_records(payload, train, {'classes': {'a': [1, 2]}})


def test_loads_a_single_extracted_receiver_file(tmp_path):
    path = tmp_path / 'receiver.pkl'
    with path.open('wb') as file:
        pickle.dump({'node_list': ['a'], 'data': [[1, 2]]}, file)
    assert load_source({'archive': str(path), 'member': None})['node_list'] == ['a']
