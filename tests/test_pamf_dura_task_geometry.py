import torch
from experiments.pamf_dura_efficiency.task_geometry import decision_metrics


def test_clean_F_choice_margin_is_teacher_fidelity_not_unknown_truth():
    reference=torch.tensor([[3.,1.,0.,0.,0.,0.],[0.,1.,4.,0.,0.,0.]])
    actual=torch.tensor([[1.,2.,0.,0.,0.,0.],[0.,1.,2.,0.,0.,0.]])
    values=decision_metrics(actual,reference)
    torch.testing.assert_close(values['teacher_argmax_agreement'],torch.tensor([0.,1.]))
    torch.testing.assert_close(values['margin_for_clean_F_choice'],torch.tensor([-1.,1.]))
    torch.testing.assert_close(values['clean_F_margin'],torch.tensor([2.,3.]))
