import copy
import pytest
import torch
from torch import nn
from tools.r17_task_fidelity import task_terms,frozen_task_fidelity


def test_matching_reference_is_zero_and_teacher_values_never_receive_gradients():
    torch.manual_seed(1)
    features=torch.randn(3,160,requires_grad=True);logits=torch.randn(3,6,requires_grad=True)
    desired_features=features.detach().clone().requires_grad_();desired_logits=logits.detach().clone().requires_grad_()
    loss,terms=task_terms(features,logits,desired_features,desired_logits)
    assert torch.equal(loss,torch.zeros_like(loss))
    loss.mean().backward()
    assert desired_features.grad is None and desired_logits.grad is None
    assert features.grad is not None and features.grad.abs().max()<1e-7
    assert logits.grad is not None and logits.grad.abs().max()<1e-7


def test_feature_scale_and_common_logit_offset_do_not_create_task_error():
    torch.manual_seed(2)
    features=torch.randn(2,160);logits=torch.randn(2,6)
    loss,_=task_terms(features*2,logits+5,features,logits)
    assert loss.abs().max()<1e-6


def test_changed_posterior_and_feature_directions_backpropagate_without_truth_labels():
    torch.manual_seed(3)
    desired_features=torch.randn(2,160,requires_grad=True);desired_logits=torch.randn(2,6,requires_grad=True)
    features=(desired_features.detach()+.1*torch.randn(2,160)).requires_grad_()
    logits=(desired_logits.detach()+.5*torch.randn(2,6)).requires_grad_()
    loss,terms=task_terms(features,logits,desired_features,desired_logits)
    assert loss.gt(0).all() and torch.equal(loss,terms['task_KL']+terms['task_feature_distance'])
    loss.mean().backward()
    assert features.grad.abs().sum()>0 and logits.grad.abs().sum()>0
    assert desired_features.grad is None and desired_logits.grad is None


class TinyReader(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=nn.Linear(512,160);self.head=nn.Linear(160,6);self.calls=0
    def features(self,x):self.calls+=1;return self.encoder(x.flatten(1)).tanh()
    def classify_features(self,x):return self.head(x)


def test_frozen_existing_head_allows_input_gradients_without_parameter_updates():
    torch.manual_seed(4)
    reader=TinyReader().eval().requires_grad_(False);before=copy.deepcopy(reader.state_dict())
    teacher=torch.randn(2,2,256,requires_grad=True);student=(teacher.detach()+.1*torch.randn(2,2,256)).requires_grad_()
    loss,_=frozen_task_fidelity(student,teacher,reader);loss.mean().backward()
    assert student.grad is not None and student.grad.abs().sum()>0 and teacher.grad is None
    assert reader.calls==2
    assert all(p.grad is None for p in reader.parameters())
    assert all(torch.equal(value,before[key]) for key,value in reader.state_dict().items())


@pytest.mark.parametrize('train_mode,trainable',[(True,False),(False,True)])
def test_trainable_or_training_reader_rejected_before_forward(train_mode,trainable):
    reader=TinyReader().train(train_mode).requires_grad_(trainable);x=torch.ones(2,2,256,requires_grad=True)
    with pytest.raises(ValueError):frozen_task_fidelity(x,x.detach(),reader)
    assert reader.calls==0


def test_wrong_feature_or_IQ_shapes_rejected():
    with pytest.raises(ValueError):task_terms(torch.ones(2,159),torch.ones(2,6),torch.ones(2,159),torch.ones(2,6))
    reader=TinyReader().eval().requires_grad_(False)
    with pytest.raises(ValueError):frozen_task_fidelity(torch.ones(2,2,255),torch.ones(2,2,255),reader)
    assert reader.calls==0
