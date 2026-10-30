import copy
import math
import numpy as np
import pytest
import torch

from experiments.pamf_phase2_practical.features import tensor
from experiments.pamf_phase2_practical import external_additional as extra
from experiments.pamf_phase2_practical.external_acceleration import checked_step, issl_ssl, issl_transfer
from experiments.pamf_phase2_practical.vendor_native.comparison_suite.adaptation import _grad_norm, _batch_scalars
from experiments.pamf_phase2_practical.vendor_native.comparison_suite import adaptation


@pytest.mark.parametrize('kind', ['normal','noncontiguous','readonly','empty','scalar'])
def test_buffer_tensor_preserves_values_and_independence(kind):
    x=np.arange(48,dtype=np.float32).reshape(3,2,8)
    if kind=='noncontiguous':x=x[...,::2]
    if kind=='readonly':x.setflags(write=False)
    if kind=='empty':x=x[:0]
    if kind=='scalar':x=np.asarray(-0.,dtype=np.float32)
    original=torch.tensor(np.asarray(x,dtype=np.float32).tolist(),device='cpu',dtype=torch.float32)
    result=tensor(x,'cpu')
    assert result.shape==original.shape
    assert torch.equal(result,original)
    if result.numel() and kind not in ('readonly','scalar'):
        result.flatten()[0]=999
        assert np.asarray(x).flat[0]!=999


@pytest.mark.parametrize('bad', [float('nan'),float('inf'),float('-inf')])
def test_fast_step_rejects_nonfinite_gradients(bad):
    p=torch.nn.Parameter(torch.tensor([1.]))
    opt=torch.optim.SGD([p],lr=.1)
    p.register_hook(lambda grad:grad*bad)
    with pytest.raises(FloatingPointError,match='gradient'):
        checked_step(p.square().sum(),opt)
    assert p.item()==1.


def test_gradient_norm_and_scalar_conversion_match_original():
    m=torch.nn.Sequential(torch.nn.Linear(3,7),torch.nn.Linear(7,2))
    m(torch.ones(4,3)).sum().backward()
    old=math.sqrt(sum(float(p.grad.detach().square().sum()) for p in m.parameters() if p.grad is not None))
    assert _grad_norm(m)==old
    values={'loss':torch.tensor(1.25),'other':torch.tensor(3.5),'iteration':4}
    assert _batch_scalars(values)=={'loss':1.25,'other':3.5,'iteration':4}


def test_radionet_updates_and_metrics_match_original(monkeypatch):
    torch.set_num_threads(1)
    torch.manual_seed(123)
    base=adaptation.RadioNetADANet(6)
    discriminator=adaptation.RadioNetADAHead(1)
    models=[copy.deepcopy(base),copy.deepcopy(base)]
    discs=[copy.deepcopy(discriminator),copy.deepcopy(discriminator)]
    x=torch.randn(8,2,256);y=torch.arange(8)%6;target=torch.randn(8,2,256)
    fast_norm=adaptation._grad_norm
    def old_norm(model):
        return math.sqrt(sum(float(p.grad.detach().square().sum()) for p in model.parameters() if p.grad is not None))
    history=[]
    for i in range(2):
        monkeypatch.setattr(adaptation,'_grad_norm',old_norm if i==0 else fast_norm)
        optim=torch.optim.Adam(models[i].parameters(),lr=.00001)
        optim_d=torch.optim.Adam(discs[i].parameters(),lr=.00001)
        torch.manual_seed(81);rows=[]
        for _ in range(3):
            terms=adaptation.radionet_ada_batch_step(models[i],discs[i],x,y,target,optim,optim_d)
            rows.append({k:adaptation._scalar(v) for k,v in terms.items()} if i==0 else adaptation._batch_scalars(terms))
        history.append(rows)
    for a,b in ((models[0],models[1]),(discs[0],discs[1])):
        for name,value in a.state_dict().items():
            torch.testing.assert_close(b.state_dict()[name],value,rtol=0,atol=0,msg=name)
    assert history[0]==history[1]


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_issl_author_training_and_transfer_match_original():
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic=True
    extra._seed(901)
    original=extra.NativeModel('issl_author_behavior',classes=6).cuda()
    accelerated=copy.deepcopy(original)
    g=torch.Generator().manual_seed(731)
    x=torch.randn(65,2,256,generator=g)
    y=torch.randint(0,6,(65,),generator=g)
    spec={'C_SSL_epochs':2,'C_SSL_lr':1e-5}
    old_log=[];new_log=[]
    extra._seed(42);extra._issl_ssl(original,x,6,spec,42,old_log.append)
    extra._seed(42);issl_ssl(accelerated,x,6,spec,42,new_log.append)
    for name,value in original.state_dict().items():
        torch.testing.assert_close(accelerated.state_dict()[name],value,rtol=0,atol=0,msg=name)
    assert old_log==new_log
    old_log=[];new_log=[]
    extra._seed(51);extra._train_ce(original,x,y,2,32,.03,51,old_log.append,crop=218,stage='C_transfer')
    extra._seed(51);issl_transfer(accelerated,x,y,2,32,.03,51,new_log.append,crop=218)
    for name,value in original.state_dict().items():
        torch.testing.assert_close(accelerated.state_dict()[name],value,rtol=0,atol=0,msg=name)
    assert old_log==new_log
