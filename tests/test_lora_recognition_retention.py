import torch

from experiments.pamf_lora_main.dura_real_gain_lora_retention import recognition_retention


def test_retention_penalizes_harming_source_teacher_and_hides_masked_labels():
    teacher = torch.tensor([[4., 0.], [1., 2.]])
    labels = torch.tensor([0, 0])
    mask = torch.tensor([True, False])
    same = recognition_retention(teacher, teacher, labels, mask)
    changed_ignored = teacher.clone()
    changed_ignored[1] = torch.tensor([-20., 20.])
    torch.testing.assert_close(recognition_retention(changed_ignored, teacher, labels, mask), same)
    harmed = teacher.clone()
    harmed[0] = torch.tensor([0., 4.])
    assert recognition_retention(harmed, teacher, labels, mask) > same
