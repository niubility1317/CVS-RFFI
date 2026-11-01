import numpy as np
import torch
from phase1_blind_gate_model import BlindGate
from phase1_blind_gate_predict import apply
from phase1_blind_gate_source import gate_loss,choose
from phase1_axial_spectral_mask import AxialSpectralMask

def main():
    torch.set_num_threads(2);torch.manual_seed(4);gate=BlindGate();x=torch.randn(4,2,256);loss=gate_loss(gate,x,np.array(['clean','clean','practical_high','practical_low_urban']));loss.backward()
    assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in gate.parameters())
    model=AxialSpectralMask();y,active=apply(model,gate,.5,1.,x);assert not active.any() and torch.equal(y,x)
    a=dict(count=100,valid_count=100,active=10,raw_correct=60,candidate_correct=72,raw_iq=.1,candidate_iq=.09)
    b=dict(count=100,valid_count=100,active=0,raw_correct=100,candidate_correct=100,raw_iq=0.,candidate_iq=0.)
    h=dict(count=100,valid_count=100,active=10,raw_correct=90,candidate_correct=0,raw_iq=.1,candidate_iq=1.)
    cells={'1:4:clean':b,'1:4:practical_low_urban':a,'2:8:practical_low_urban':h};sel=choose({1:{'0.25:0.8':cells}})
    assert sel['kind']=='gated_direct_iq' and sel['validation_net']==12
    cells['1:4:clean']['candidate_iq']=.002;assert choose({1:{'0.25:0.8':cells}})['kind']=='raw'
    print('VERIFIED gate gradient, exact disabled identity and validation-only protection')
if __name__=='__main__':main()
