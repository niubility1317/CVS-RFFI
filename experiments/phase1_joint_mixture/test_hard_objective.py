import torch
from hard_objective import hard_objective

def test_hard_sample_has_fivefold_classification_gradient():
 logits=torch.zeros(2,6,requires_grad=True);controller=torch.zeros(2,6,requires_grad=True);ground=torch.ones(2,2,256);predicted=ground.clone();target=torch.tensor([0,0]);hard=torch.tensor([False,True]);loss=hard_objective(logits,controller,predicted,ground,target,hard);loss.backward();assert torch.allclose(logits.grad[1],5*logits.grad[0]);assert torch.allclose(controller.grad[1],5*controller.grad[0])

def test_misaligned_hard_mask_rejected():
 try:hard_objective(torch.zeros(2,6),torch.zeros(2,6),torch.ones(2,2,256),torch.ones(2,2,256),torch.tensor([0,1]),torch.tensor([True]))
 except ValueError:return
 raise AssertionError('incorrect hard mask accepted')
