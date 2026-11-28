import torch

from experiments.pamf_lora_main.dura_real_gain_wisig_consistency import consistency_loss


def test_clean_r17_teacher_and_retention_define_the_training_target():
    teacher = torch.ones(2, 256, dtype=torch.complex64)
    equal = consistency_loss(teacher, teacher, teacher, teacher)
    assert equal.item() == 0
    changed = teacher.clone()
    changed[:, 4] += .5
    loss = consistency_loss(changed, teacher, teacher, teacher)
    assert torch.isfinite(loss) and loss > 0
