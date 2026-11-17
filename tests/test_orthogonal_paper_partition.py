import pytest
from experiments.orthogonal_paper_repro.partition import build_partition


def test_paper_wisig_partition_is_deterministic_and_disjoint():
    counts = {f'tx-{i:03d}': 50 + i % 9 for i in range(140)}
    training, evaluation = build_partition(counts, seed=2026092701)
    assert (training, evaluation) == build_partition(counts, seed=2026092701)
    assert len(training['base']) == 60
    assert len(training['increments']) == 7
    assert all(len(session) == 10 for session in training['increments'])
    assert len(evaluation['classes']) == 130
    assert set(training).isdisjoint({'evaluation', 'query', 'test'})
    for cls in training['base']:
        test_ids = evaluation['classes'][cls['id']]
        assert len(cls['indices']) == int(.8 * counts[cls['id']])
        assert set(cls['indices']).isdisjoint(test_ids)
        assert len(cls['indices']) + len(test_ids) == counts[cls['id']]
    for session in training['increments']:
        for cls in session:
            test_ids = evaluation['classes'][cls['id']]
            assert len(cls['indices']) == 5
            assert set(cls['indices']).isdisjoint(test_ids)
            assert len(test_ids) + 5 == counts[cls['id']]


def test_insufficient_classes_are_rejected():
    with pytest.raises(ValueError):
        build_partition({str(i): 50 for i in range(129)}, seed=1)
