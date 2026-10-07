from types import SimpleNamespace
import pytest
import torch
from experiments.cvs_phase1_repair.single_view import SingleViewLEO,single_view_loss


class Renderer:
    def __init__(self): self.seen=[]
    def __call__(self,x,batch,epoch,step):
        self.seen.append(batch)
        assert all(set(m)=={'sample_id','rx_i','day_i'} for m in batch['meta'])
        return SimpleNamespace(x=x+1,applied=True,scenario='synthetic',view_prob=1.)


def batch(n):
    return dict(meta=[dict(sample_id=str(i),rx_i=i%2,day_i=1,tx_i=i%6) for i in range(n)])


def test_early_epochs_skip_renderer_and_keep_input():
    r=Renderer();a=SingleViewLEO(None,seed=1,renderer=r);x=torch.zeros(32,2,256)
    z,info=a(x,batch(32),79,1)
    assert z is x and not r.seen and info['loss_scale']==1


def test_single_model_call_and_metadata_alignment():
    r=Renderer();a=SingleViewLEO(None,seed=3,renderer=r);x=torch.zeros(128,2,256)
    y=torch.arange(128)%6;domain=torch.arange(128)%2;calls=[]
    def model(value,**kw):
        calls.append(value.clone());assert torch.equal(kw['y'],y)
        assert torch.equal(kw['domain_labels'],domain)
        return torch.zeros(128,6,requires_grad=True)
    loss,info=single_view_loss(model,x,y,domain,batch(128),a,100,1)
    loss.backward()
    assert len(calls)==1 and calls[0].shape==x.shape and x.sum()==0
    ids=[int(m['sample_id']) for m in r.seen[0]['meta']]
    assert set(calls[0].sum((1,2)).nonzero().flatten().tolist())==set(ids)
    assert info['model_samples']==128 and info['rendered_samples']==len(ids)
    assert float(loss.detach())==pytest.approx(1.68*torch.log(torch.tensor(6.)).item())


def test_weighted_objective_identity_and_replay():
    # Exact expectation for arbitrary clean/augmented per-item losses.
    q=.68/1.68;clean=torch.tensor([.2,1.3]);sat=torch.tensor([1.2,.9])
    assert torch.allclose(1.68*((1-q)*clean+q*sat),clean+.68*sat)
    x=torch.zeros(128,2,256)
    a=SingleViewLEO(None,seed=29,renderer=Renderer());b=SingleViewLEO(None,seed=29,renderer=Renderer())
    for e in (80,91,200):
        za,ia=a(x,batch(128),e,1);zb,ib=b(x,batch(128),e,1)
        assert torch.equal(za,zb) and ia==ib
