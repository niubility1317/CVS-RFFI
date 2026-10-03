import io
import numpy as np
import pytest
import torch
from experiments.cvs_reference_residual_identity.model import ReferenceResidual,build,VARIANTS
from experiments.cvs_reference_residual_identity.physics import signals
from experiments.cvs_residual_identity.model import build as base

@pytest.mark.parametrize('variant',VARIANTS)
def test_initial_function_and_ce_gradient(variant):
    torch.set_num_threads(2);torch.manual_seed(54);model=build(variant).eval();torch.manual_seed(54);original=base('residual_fusion').eval()
    x=torch.randn(4,2,256);y=torch.tensor([0,1,2,3])
    with torch.no_grad():torch.testing.assert_close(model(x),original(x),atol=0,rtol=0)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4)
    for step in range(3):
        optimizer.zero_grad();loss=torch.nn.functional.cross_entropy(model(x),y);loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
    assert model.response.ridge_raw.grad.abs().max()>0
    buffer=io.BytesIO();torch.save(model.state_dict(),buffer);buffer.seek(0)
    restored=build(variant).eval();restored.load_state_dict(torch.load(buffer,weights_only=True))
    with torch.no_grad():torch.testing.assert_close(model(x),restored(x),atol=0,rtol=0)
    assert all(not v.is_floating_point() or v.dtype==torch.float32 for v in model.state_dict().values())

@pytest.mark.parametrize('variant',VARIANTS)
def test_zero_small_and_phase_gain(variant):
    layer=ReferenceResidual(variant);torch.manual_seed(123);x=torch.randn(3,2,256)
    for scale in (0.,1e-12,1.,100.):
        a=(x*scale).requires_grad_();z=layer(a);z.square().mean().backward();assert torch.isfinite(z).all() and torch.isfinite(a.grad).all()
    c,s=np.cos(.4),np.sin(.4);rot=3*torch.stack((c*x[:,0]-s*x[:,1],s*x[:,0]+c*x[:,1]),1)
    with torch.no_grad():torch.testing.assert_close(layer(x),layer(rot),rtol=5e-4,atol=3e-5)

def test_public_continuous_cascade_and_matrix():
    records,x=signals();assert len(records)==72 and len({tuple(r.values()) for r in records})==72
    assert np.isfinite(x).all();np.testing.assert_allclose((x*x).sum(1).mean(1),1,atol=2e-7)
    # Pure repeated template has no missing-history transient: delay16 uses analytic n-16.
    from experiments.cvs_reference_residual_identity.physics import transmit
    z=transmit(np.arange(256),'ideal')+(.23+.09j)*transmit(np.arange(256)-16,'ideal');z/=np.sqrt(np.mean(abs(z)**2))
    index=next(i for i,r in enumerate(records) if r==dict(tx='ideal',channel='delay16',rx='identity'))
    np.testing.assert_array_equal(x[index],np.stack((z.real,z.imag)).astype(np.float32))

def test_both_arms_same_parameters_and_ridge_bounds():
    counts=[]
    for variant in VARIANTS:
        model=build(variant);counts.append(model.contract()['total_parameters']);p=model.response.ridge_raw
        with torch.no_grad():p.fill_(-100)
        assert torch.all(model.response.ridge()>=1e-4)
        with torch.no_grad():p.fill_(100)
        assert torch.all(model.response.ridge()<=.1001)
    assert counts[0]==counts[1]

@pytest.mark.parametrize('variant',VARIANTS)
def test_independent_numpy_operator(variant):
    from experiments.cvs_reference_residual_identity.recount import numpy_components
    # Public synthetic noise fixture, not the formal TX/channel probe matrix.
    x=np.random.default_rng(81).normal(size=(5,2,256)).astype(np.float32);model=ReferenceResidual(variant)
    with torch.no_grad():actual=model.components(torch.from_numpy(x))
    expected=numpy_components(x,variant,model.ridge().detach().numpy())
    for key,value in actual.items():np.testing.assert_allclose(value.numpy(),expected[key],rtol=1e-6 if key=='features' else 1e-9,atol=1e-7 if key=='features' else 1e-9)

@pytest.mark.parametrize('tamper',[None,'missing_pair','wrong_drift','duplicate_input'])
def test_complete_public_recount(tmp_path,monkeypatch,tamper):
    import json
    from experiments.cvs_reference_residual_identity import physics,recount
    output=tmp_path/'fixture';monkeypatch.setattr(physics,'OUT',output);monkeypatch.setattr(recount,'OUT',output)
    physics.run()
    if tamper in ('missing_pair','wrong_drift'):
        p=output/'metrics.json';d=json.loads(p.read_text())
        if tamper=='missing_pair':d['rows'].pop()
        else:d['rows'][-1]['relative_signature_drift']+=.1
        p.write_text(json.dumps(d),encoding='utf-8')
    if tamper=='duplicate_input':
        p=output/'records.json';d=json.loads(p.read_text());d[-1]=d[0];p.write_text(json.dumps(d),encoding='utf-8')
    if tamper:
        with pytest.raises(ValueError):recount.recount()
        assert not (output/'independent_recount.json').exists()
    else:
        recount.recount();assert json.loads((output/'independent_recount.json').read_text())['status']=='VERIFIED'
        with pytest.raises(FileExistsError):recount.recount()
