import pytest
import torch
from experiments.pamf_consensus_preserve.model import select_iq


def test_restore_requires_both_fixed_signals():
    x=torch.ones(4,2,256);r=x.clone();r[:2]+=.01;r[2:]+=1
    logits=torch.tensor([-1.,1.,-1.,1.]);value,keep=select_iq(x,r,logits)
    assert keep.tolist()==[True,True,True,False]
    assert torch.equal(value[:3],x[:3]) and torch.equal(value[3],r[3])
    assert torch.equal(value,torch.cat([select_iq(x[i:i+1],r[i:i+1],logits[i:i+1])[0] for i in range(4)]))


@pytest.mark.parametrize('malformation',['shape','logits','nonfinite'])
def test_invalid_inputs_fail(malformation):
    x=torch.ones(2,2,256);r=x.clone();logits=torch.ones(2)
    if malformation=='shape':r=r[:1]
    elif malformation=='logits':logits=torch.ones(2,1)
    else:r[0,0,0]=float('nan')
    with pytest.raises((ValueError,FloatingPointError)):select_iq(x,r,logits)
