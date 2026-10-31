import torch
from torch import nn

from experiments.pamf_phase1_flow.feature_conditioned import FeatureConditionedCorrector, source_targets, weighted_margin
from experiments.pamf_phase1_flow.feature_conditioned_experiment import step


def test_source_targets_use_legal_labels_and_confident_pseudo_labels_only():
    clean_logits=torch.tensor([[0.,2.],[3.,0.],[0.,0.]])
    labels=torch.tensor([-1,-1,1])
    target,valid=source_targets(labels,clean_logits,threshold=0.95)
    assert valid.tolist()==[False,True,True]
    assert target[1:].tolist()==[0,1]


def test_margin_pushes_toward_target_and_ignores_untrusted_pseudo():
    logits=torch.tensor([[0.,1.],[0.,1.]],requires_grad=True)
    loss=weighted_margin(logits,torch.tensor([0,0]),torch.tensor([True,False]),torch.tensor([1,1]))
    loss.backward()
    assert logits.grad[0,0]<0 and logits.grad[0,1]>0
    assert torch.equal(logits.grad[1],torch.zeros(2))


def test_feature_conditioned_correction_starts_as_anchor_and_is_bounded():
    model=FeatureConditionedCorrector(feature_dim=160,classes=6,width=16,layers=1,heads=4,epsilon=0.1).eval()
    x=torch.randn(3,2,256);anchor=torch.randn_like(x);anchor[0]=0
    features=torch.randn(3,160);logits=torch.randn(3,6)
    assert torch.equal(model(x,anchor,features,logits),anchor)
    with torch.no_grad():model.output.bias.fill_(10.)
    result=model(x,anchor,features,logits)
    assert torch.all((result-anchor).flatten(1).norm(dim=1)<=0.1*anchor.flatten(1).norm(dim=1)+1e-5)
    assert torch.equal(result[0],anchor[0])


def test_source_step_updates_only_student_using_labeled_targets():
    class Teacher(nn.Module):
        def features(self,x):return x.flatten(1).mean(1,keepdim=True)
        def classify_features(self,x):return torch.cat((x,-x),dim=1)
    class Student(nn.Module):
        def __init__(self):super().__init__();self.value=nn.Parameter(torch.tensor(0.))
        def forward(self,received,anchor,features,logits):return anchor+self.value
    teacher=Teacher().eval().requires_grad_(False);dura=nn.Identity().eval().requires_grad_(False)
    student=Student();optimizer=torch.optim.SGD(student.parameters(),lr=0.1)
    stats=step(student,teacher,dura,-torch.ones(2,2,256),torch.ones(2,2,256),torch.tensor([0,0]),optimizer)
    assert stats['labeled']==2 and stats['grad_norm']>0 and student.value.item()>0
