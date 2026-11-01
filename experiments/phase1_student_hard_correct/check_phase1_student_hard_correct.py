import torch
from torch import nn
from phase1_axial_spectral_mask import AxialSpectralMask
from phase1_blind_gate_model import BlindGate
from phase1_blind_gate_predict import apply
from phase1_teacher_student_source import objective,curriculum_weights,choose

class Teacher(nn.Module):
    def __init__(self):
        super().__init__();self.head=nn.Linear(512,6)
        with torch.no_grad():
            self.head.weight.zero_();self.head.bias.fill_(-10.);self.head.weight[0,0]=8.;self.head.weight[1,0]=-8.;self.head.bias[0]=0.;self.head.bias[1]=0.
    def features(self,x):return x.flatten(1)
    def classify_features(self,x):return self.head(x)
class Dura(nn.Module):
    def forward(self,x):return -x

def main():
    torch.manual_seed(81);torch.set_num_threads(2);model=AxialSpectralMask();teacher=Teacher().eval().requires_grad_(False);dura=Dura().eval().requires_grad_(False)
    x=torch.ones(2,2,256);y=x.clone();y[1]=-x[1]
    loss,hard=objective(model,dura,teacher,x,y);assert hard.tolist()==[True,False]
    iw,cw=curriculum_weights(hard);torch.testing.assert_close(iw,torch.tensor([.25,1.]));torch.testing.assert_close(cw,torch.tensor([3.,.2]))
    loss.backward();assert torch.isfinite(loss)
    for p in (model.phase_head[2].weight,model.temporal_output.weight,model.output.weight):assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
    assert all(p.grad is None and not p.requires_grad for p in teacher.parameters())
    gate=BlindGate();out,active=apply(model,gate,.5,1.,x);assert not active.any() and torch.equal(out,x)
    a=dict(count=100,valid_count=100,active=99,baseline_correct=60,candidate_correct=72,baseline_iq=.1,candidate_iq=.09);b=dict(count=100,valid_count=100,active=0,baseline_correct=100,candidate_correct=100,baseline_iq=.00001,candidate_iq=0.)
    cells={'1:4:clean':b,'1:4:practical_low_urban':a};assert choose({1:{'0.5:0.8':cells}})['validation_net']==12
    cells['1:4:clean']['candidate_iq']=.002;assert choose({1:{'0.5:0.8':cells}})['kind']=='dura'
    print('VERIFIED teacher-error curriculum, frozen gradients and DURA-relative selection')
if __name__=='__main__':main()
