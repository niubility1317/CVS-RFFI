import torch
from torch import nn

from experiments.pamf_phase1_flow.boundary_student import BoundedBoundaryStudent, distillation_loss, make_source_oracle
from experiments.pamf_phase1_flow.boundary_student_experiment import step


class ConstantField(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(float(value)))

    def velocity(self, anchor, received, time):
        return torch.ones_like(anchor) * self.value


def test_student_uses_received_and_anchor_only_and_limits_relative_change():
    student = BoundedBoundaryStudent(ConstantField(3.0), epsilon=0.05)
    received = torch.randn(3, 2, 256)
    anchor = torch.randn(3, 2, 256)
    anchor[0] = 0
    result = student(received, anchor)
    change = (result - anchor).flatten(1).norm(dim=1)
    assert torch.all(change <= 0.05 * anchor.flatten(1).norm(dim=1) + 1e-5)
    assert torch.equal(result[0], anchor[0])


def test_distillation_emphasizes_teacher_disagreements_without_using_tx_labels():
    anchor = torch.ones(2, 2, 4)
    student = anchor.clone()
    oracle = anchor.clone()
    oracle[1] += 0.5
    wrong = torch.tensor([False, True])
    loss = distillation_loss(student, oracle, anchor, wrong, hard_weight=8.0)
    assert torch.isclose(loss, torch.tensor(2.0 / 9.0))


def test_zero_field_starts_as_exact_dura_anchor():
    student = BoundedBoundaryStudent(ConstantField(0.0), epsilon=0.05)
    x = torch.randn(2, 2, 256)
    anchor = torch.randn_like(x)
    assert torch.equal(student(x, anchor), anchor)


def test_privileged_training_oracle_changes_only_teacher_disagreements():
    class Teacher(nn.Module):
        def features(self, x):
            return x.flatten(1).mean(1, keepdim=True)
        def classify_features(self, x):
            return torch.cat((x, -x), dim=1)
    teacher = Teacher().eval().requires_grad_(False)
    anchor = torch.stack((torch.ones(2, 256), -torch.ones(2, 256)))
    clean = torch.ones_like(anchor)
    result, wrong = make_source_oracle(teacher, anchor, clean, epsilon=0.05, steps=4)
    assert wrong.tolist() == [False, True]
    assert torch.equal(result[0], anchor[0])
    assert result[1].mean() > anchor[1].mean()
    assert not result.requires_grad


def test_source_step_trains_only_student_on_real_disagreements():
    class Teacher(nn.Module):
        def features(self, x):
            return x.flatten(1).mean(1, keepdim=True)
        def classify_features(self, x):
            return torch.cat((x, -x), dim=1)
    teacher = Teacher().eval().requires_grad_(False)
    dura = nn.Identity().eval().requires_grad_(False)
    student = BoundedBoundaryStudent(ConstantField(0.0), epsilon=0.05)
    optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
    received = -torch.ones(2, 2, 256)
    stats = step(student, teacher, dura, received, torch.ones_like(received), optimizer)
    assert stats['wrong'] == 2 and stats['grad_norm'] > 0
    assert student.field.value.item() > 0
