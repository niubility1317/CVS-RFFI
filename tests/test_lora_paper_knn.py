import numpy as np
import pytest
from experiments.lora_paper_repro.knn import classify


def test_author_fifteen_neighbor_rule_uses_enrollment_labels_only():
    gallery = np.r_[np.zeros((15, 2)), np.ones((15, 2)) * 4]
    labels = np.repeat([30, 31], 15)
    predictions = classify(gallery, labels, np.array([[.1, .1], [3.9, 4.1]]))
    np.testing.assert_array_equal(predictions, [30, 31])


def test_fifteen_neighbors_cannot_exceed_gallery_size():
    with pytest.raises(ValueError):
        classify(np.zeros((14, 2)), np.zeros(14, dtype=int), np.zeros((1, 2)))
