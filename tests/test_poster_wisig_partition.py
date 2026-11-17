import numpy as np
from experiments.poster_paper_repro.wisig_partition import select_indices


def test_cross_day_selection_is_fixed_paired_and_disjoint():
    counts1 = {f'd{i}': 300 for i in range(6)}
    counts2 = {f'd{i}': 250 for i in range(6)}
    source, target, truth = select_indices(counts1, counts2, seed=22)
    assert len(source) == len(target) == len(truth) == 5
    for source_row, target_row, query_row in zip(source, target, truth):
        assert source_row['id'] == target_row['id'] == query_row['id']
        assert len(source_row['train']) == 200 and len(source_row['validation']) == 40
        assert not set(source_row['train']) & set(source_row['validation'])
        assert len(target_row['support']) == 10 and len(query_row['query']) == 100
        assert not set(target_row['support']) & set(query_row['query'])
    assert (source, target, truth) == select_indices(counts1, counts2, seed=22)
