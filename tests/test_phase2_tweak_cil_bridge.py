import torch
from torch import nn

from experiments.pamf_phase2_practical import external_additional as extra
from experiments.pamf_phase2_practical import group2_cil_bridge as bridge
from experiments.pamf_phase2_practical import group2_cross_fit


class ToyTweak(nn.Module):
    def __init__(self):
        super().__init__();self.profile='tweak_m1';self.layer=nn.Linear(2,2,bias=False)
        with torch.no_grad():self.layer.weight.copy_(torch.eye(2))

    def raw(self,x):return self.layer(x)


def test_tweak_bridge_preserves_native_m1_decisions_and_gradients():
    model=ToyTweak().eval();x=torch.tensor([[0.,0.],[2.,0.],[0.,3.],[3.,4.]])
    labels=torch.tensor([0,0,1,1]);state=dict(profile='tweak_m1',resources={},native_state={})
    state=extra._metric_state(state,extra.native_features(model,x),labels)
    adapter=bridge.from_tweak(model,state,old_count=2)
    q=torch.tensor([[0.1,0.],[2.5,1.],[7.,8.]],requires_grad=True)
    native=state['score_fn'](model,q)
    actual=adapter(q)['tx_logits']
    assert torch.equal(actual.argmax(1),native.argmax(1))
    assert torch.isfinite(adapter(q)['features']).all() and adapter(q)['features'].abs().max()<100
    assert torch.allclose(torch.cat([adapter(z[None])['tx_logits'] for z in q]),actual)
    actual.sum().backward();assert torch.isfinite(q.grad).all()


def test_tweak_creates_two_c_anchors_from_same_b():
    model=ToyTweak().eval();x=torch.tensor([[0.,0.],[2.,0.],[0.,3.],[3.,4.]])
    labels=torch.tensor([0,0,1,1]);state=dict(profile='tweak_m1',resources={},native_state={})
    state=extra._metric_state(state,extra.native_features(model,x),labels)
    prepared=group2_cross_fit.prepare_b(model,'tweak_m1',extra.native_features,x,labels,old_count=2,b_scores=state)
    assert set(prepared)=={'lora','mopc_hr','orthogonal'}
    assert prepared['mopc_hr'].classes==prepared['orthogonal'].classes==(0,1)
    assert torch.equal(prepared['mopc_hr'].prototypes,prepared['orthogonal'].prototypes)
    new_x=torch.tensor([[4.,0.],[5.,0.]])
    new_y=torch.tensor([2,2])
    result=bridge.register(prepared['orthogonal'],new_x,new_y,seed=7,epochs=1)
    assert result.classes==(0,1,2)
    assert torch.isfinite(result.model(new_x)['tx_logits']).all()
    mopc=bridge.register(prepared['mopc_hr'],new_x,new_y,seed=7,epochs=1)
    assert mopc.classes==(0,1,2)
    assert torch.isfinite(mopc.model(new_x)['tx_logits']).all()


def test_native_tweak_encoder_support_only_mopc_bridge():
    torch.manual_seed(4);torch.set_num_threads(1)
    model=extra.NativeModel('tweak_m1',6,5).eval()
    x=torch.randn(12,2,256);y=torch.arange(6).repeat_interleave(2)
    state=extra._metric_state(dict(profile='tweak_m1',resources={},native_state={}),
                              extra.native_features(model,x),y)
    prepared=group2_cross_fit.prepare_b(model,'tweak_m1',extra.native_features,x,y,old_count=6,b_scores=state)
    new_x=torch.randn(4,2,256);new_y=torch.tensor([6,6,7,7])
    result=bridge.register(prepared['mopc_hr'],new_x,new_y,seed=7,epochs=1)
    assert result.classes==tuple(range(8))
    assert torch.isfinite(result.model(new_x)['tx_logits']).all()
