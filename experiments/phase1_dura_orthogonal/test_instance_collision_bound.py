import itertools
import pytest

from experiments.phase1_dura_orthogonal.instance_collision_bound import minimum_same_class_ordered_pairs


def test_exact_small_cases_against_all_class_assignments():
    for n in range(1, 7):
        for classes in range(1, 4):
            values = []
            for assignment in itertools.product(range(classes), repeat=n):
                values.append(sum(assignment[i] == assignment[j] for i in range(n) for j in range(n) if i != j))
            assert minimum_same_class_ordered_pairs(n, classes) == min(values)


def test_32_unique_records_in_six_classes():
    assert minimum_same_class_ordered_pairs(32, 6) == 140


def test_invalid_arguments_rejected():
    with pytest.raises(ValueError):
        minimum_same_class_ordered_pairs(-1, 6)
    with pytest.raises(ValueError):
        minimum_same_class_ordered_pairs(32, 0)
