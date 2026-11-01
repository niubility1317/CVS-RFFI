import torch
from torch import nn
from phase1_axial_spectral_mask import AxialSpectralMask
from phase1_axial_spectral_source import objective,choose

class FrozenTeacher(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=nn.Linear(512,160);self.head=nn.Linear(160,6)
    def features(self,x):return self.encoder(x.flatten(1))
    def classify_features(self,x):return self.head(x)

def main():
    torch.manual_seed(72);torch.set_num_threads(2)
    teacher=FrozenTeacher().eval().requires_grad_(False);model=AxialSpectralMask();x=torch.randn(2,2,256);y=torch.randn(2,2,256)
    loss=objective(model,teacher,x,y);loss.backward()
    assert torch.isfinite(loss) and model.output.weight.grad.abs().sum()>0 and model.temporal_output.weight.grad.abs().sum()>0
    assert all(p.grad is None and not p.requires_grad for p in teacher.parameters())
    a=dict(count=10,valid_count=10,raw_correct=7,candidate_correct=[9,5,7],raw_iq=.1,candidate_iq=[.09,.09,.09])
    b=dict(count=10,valid_count=10,raw_correct=10,candidate_correct=[10,10,10],raw_iq=0.,candidate_iq=[.0001,.0001,.0001])
    c=dict(count=10,valid_count=10,raw_correct=9,candidate_correct=[0,10,10],raw_iq=.2,candidate_iq=[1.,0.,0.])
    cases={1:{'1:4:clean':b,'1:4:practical_low_urban':a,'2:8:practical_low_urban':c}}
    assert choose(cases)==dict(kind='axial_direct_iq',epoch=1,alpha=.25,validation_net=2)
    cases[1]['1:4:clean']['candidate_correct'][0]=9
    assert choose(cases)['kind']=='raw'
    print('VERIFIED frozen-teacher gradient and validation-only selection')
if __name__=='__main__':main()
