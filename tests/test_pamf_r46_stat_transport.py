import io
import numpy as np
import pytest
import torch
from torch import nn
from experiments.pamf_r46_stat_transport.normalization import StatisticGroupNorm
from experiments.pamf_r46_stat_transport.models import CompactSuccessor
from experiments.pamf_r46_stat_transport import design as d

torch.set_num_threads(2)

@pytest.mark.parametrize('shape',[(1,8,32),(3,12,256),(2,28,16)])
def test_identity_matches_original_GN_and_uses_only_each_own_sample(shape):
    torch.manual_seed(58);original=nn.GroupNorm(4,shape[1]).double();model=StatisticGroupNorm(4,shape[1]).double()
    with torch.no_grad():model.weight.copy_(original.weight);model.bias.copy_(original.bias)
    x=torch.randn(shape,dtype=torch.float64)*.7+.3
    assert torch.allclose(original(x),model(x),rtol=1e-12,atol=1e-12)
    independent=torch.cat([model(v[None]) for v in x])
    assert torch.allclose(independent,model(x),rtol=1e-12,atol=1e-12)

def test_calibrated_formula_gradients_and_strict_state_include_all_new_parameters():
    torch.manual_seed(58);model=StatisticGroupNorm(4,12).double()
    mean=dict(slope=[.7,1.2,1.4,.8],intercept=[.1,-.2,.3,-.1]);logvar=dict(slope=[.9,.6,1.2,1.1],intercept=[.2,-.1,.1,.3])
    model.set_calibration(mean,logvar);x=(torch.randn(3,12,32,dtype=torch.float64)*.8+.4).requires_grad_(True)
    z=np.array(x.detach().tolist()).reshape(3,4,-1);mu=z.mean(-1);lv=np.log(z.var(-1)+model.eps)
    center=np.asarray(mean['slope'])*mu+np.asarray(mean['intercept']);lv=np.maximum(np.asarray(logvar['slope'])*lv+np.asarray(logvar['intercept']),np.log(model.eps))
    expected=((z-center[:,:,None])*np.exp(-.5*lv)[:,:,None]).reshape(x.shape)
    np.testing.assert_allclose(np.array(model(x).detach().tolist()),expected,rtol=1e-12,atol=1e-12)
    (model(x)-torch.randn_like(x)).square().mean().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(model.stat_parameters.grad).all() and model.stat_parameters.grad.abs().sum()>0
    assert model.stat_parameters.requires_grad and model.stat_parameters.numel()==16
    stream=io.BytesIO();torch.save(model.state_dict(),stream);stream.seek(0);fresh=StatisticGroupNorm(4,12).double();fresh.load_state_dict(torch.load(stream,weights_only=True),strict=True)
    with torch.no_grad():assert torch.equal(model(x),fresh(x))

@pytest.mark.parametrize('bad',['nonfinite','shape','outside'])
def test_wrong_calibration_cannot_enter_layer(bad):
    fit=dict(slope=[1.,1.,1.,1.],intercept=[0.,0.,0.,0.]);other=dict(fit)
    if bad=='nonfinite':fit['slope']=[float('nan'),1.,1.,1.]
    elif bad=='shape':fit['intercept']=[0.]
    else:fit['slope']=[0.,1.,1.,1.]
    with pytest.raises(ValueError):StatisticGroupNorm(4,12).set_calibration(fit,other)

@pytest.mark.parametrize('arm',d.ARMS)
def test_existing_compact_graph_full_cost_counts_and_shared_prior_semantics(arm):
    model=CompactSuccessor(arm)
    assert sum(p.numel() for p in model.parameters())==276453 and sum(p.numel() for p in model.frozen.parameters())==186959
    layers=[v for v in model.modules() if isinstance(v,StatisticGroupNorm)]
    assert len(layers)==168 and sum(v.stat_parameters.numel() for v in layers)==2688 and all(v.stat_parameters.requires_grad for v in layers)
    a,b=model.prior[:2];assert a.encoder.blocks[1][0].net[0].weight is b.encoder.blocks[1][0].net[0].weight
    assert a.encoder.blocks[1][0].net[1].stat_parameters is not b.encoder.blocks[1][0].net[1].stat_parameters
    with torch.no_grad():
        x=torch.zeros(1,2,256);y=model.eval()(x);assert y.shape==x.shape and torch.isfinite(y).all()

def test_fixed_matrix_keeps_original_and_new_parameter_budgets_separate():
    cfg=d.configuration();assert cfg['statistic_parameters']==2688 and cfg['training']['new_stat_parameters_epochs']==20
    assert cfg['training']['original_parameters_cumulative_epochs']==100
    assert cfg['comparison'].startswith('Calibrated versus identity')
    assert all(cfg['calibration'][k]==0 for k in ('TX_labels_read','F_reads','V_reads','target_reads','clean_IQ_reads','optimizer_updates'))
