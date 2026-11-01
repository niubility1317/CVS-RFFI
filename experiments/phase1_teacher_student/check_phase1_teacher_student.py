import torch
from torch import nn
from phase1_axial_spectral_mask import AxialSpectralMask
from phase1_blind_gate_model import BlindGate
from phase1_blind_gate_predict import apply
from phase1_teacher_student_source import objective,choose

class Teacher(nn.Module):
    def __init__(self):super().__init__();self.encoder=nn.Linear(512,160);self.head=nn.Linear(160,6)
    def features(self,x):return self.encoder(x.flatten(1))
    def classify_features(self,x):return self.head(x)
class Dura(nn.Module):
    def forward(self,x):return torch.stack((-x[:,1],x[:,0]),1)*.9

def main():
    torch.manual_seed(51);torch.set_num_threads(2);model=AxialSpectralMask();gate=BlindGate();teacher=Teacher().eval().requires_grad_(False);dura=Dura().eval().requires_grad_(False);x=torch.randn(2,2,256);y=torch.randn(2,2,256)
    torch.testing.assert_close(model(x),x,rtol=2e-5,atol=2e-5)
    loss=objective(model,dura,teacher,x,y);loss.backward()
    for p in (model.phase_head[2].weight,model.temporal_output.weight,model.output.weight):assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
    assert all(p.grad is None and not p.requires_grad for p in teacher.parameters())
    out,active=apply(model,gate,.5,1.,x);assert not active.any() and torch.equal(out,x)
    c=dict(count=100,valid_count=100,active=99,baseline_correct=60,candidate_correct=72,baseline_iq=.1,candidate_iq=.09);z=dict(count=100,valid_count=100,active=0,baseline_correct=100,candidate_correct=100,baseline_iq=.00001,candidate_iq=0.)
    cells={'1:4:clean':z,'1:4:practical_low_urban':c};assert choose({1:{'0.5:0.8':cells}})['validation_net']==12
    cells['1:4:clean']['candidate_iq']=.002;assert choose({1:{'0.5:0.8':cells}})['kind']=='dura'
    print('VERIFIED global phase student, frozen teacher gradients and DURA-relative guard')
if __name__=='__main__':main()
