import numpy as np
import torch
from torch import nn
from phase1_axial_spectral_mask import AxialSpectralMask
from phase1_axial_spectral_source import objective,restrict_training_labels,choose

class Teacher(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=nn.Linear(512,160);self.head=nn.Linear(160,6)
    def features(self,x):return self.encoder(x.flatten(1))
    def classify_features(self,x):return self.head(x)

def main():
    roles=np.array(['L_s','U_s','L_s']);labels=np.array([2,-1,5]);training=np.array([0,1])
    np.testing.assert_array_equal(restrict_training_labels(labels,roles,training),[2,-1,-1])
    try:restrict_training_labels(np.array([2,3,5]),roles,training)
    except ValueError:pass
    else:raise AssertionError('U_s label must be rejected')
    torch.manual_seed(71);torch.set_num_threads(2);teacher=Teacher().eval().requires_grad_(False);model=AxialSpectralMask()
    loss=objective(model,teacher,torch.randn(2,2,256),torch.randn(2,2,256),torch.tensor([2,-1]));loss.backward()
    assert torch.isfinite(loss) and model.output.weight.grad.abs().sum()>0 and model.temporal_output.weight.grad.abs().sum()>0
    assert all(p.grad is None and not p.requires_grad for p in teacher.parameters())
    a=dict(count=10,valid_count=10,raw_correct=7,candidate_correct=[9,5,7],raw_iq=.1,candidate_iq=[.09,.09,.09]);b=dict(count=10,valid_count=10,raw_correct=10,candidate_correct=[10,10,10],raw_iq=0.,candidate_iq=[.0001,.0001,.0001])
    assert choose({1:{'1:4:clean':b,'1:4:practical_low_urban':a}})['alpha']==.25
    print('VERIFIED labeled-source isolation, frozen teacher gradients and selection')
if __name__=='__main__':main()
