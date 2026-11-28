import torch
from torch import nn
import numpy as np
import pytest

from experiments.pamf_lora_main.dura_real_gain_lora_joint import configure_student, select_rows


def test_joint_student_trains_packet_and_real_gain_with_tracker_frozen():
    class Student(nn.Module):
        def __init__(self):
            super().__init__()
            self.tracker = nn.Linear(2, 2)
            self.core = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))

    student = configure_student(Student())
    assert not any(p.requires_grad for p in student.tracker.parameters())
    assert all(p.requires_grad for p in student.core.parameters())
    assert not student.tracker.training
    assert not student.core.training


def test_joint_screen_excludes_both_earlier_physical_samples():
    ids = np.arange(1000)
    rx = np.r_[np.zeros(700), np.ones(300)]
    rows = select_rows(ids, rx, np.arange(60), ids[60:360])
    assert len(rows) == 300
    assert np.all(rx[rows] == 0)
    assert not np.isin(rows, np.arange(360)).any()
    np.testing.assert_array_equal(rows, select_rows(ids, rx, np.arange(60), ids[60:360]))
    with pytest.raises(ValueError):
        select_rows(ids[:600], rx[:600], np.arange(60), ids[60:360])
