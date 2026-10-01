import pytest
import torch
import torch.nn.functional as F
from experiments.cvs_attentive_identity.model import build,VARIANTS,AttentivePool
from experiments.cvs_attentive_identity.dispatch import select_source_candidate
from experiments.cvs_attentive_identity.source import validate_config
torch.set_num_threads(2)

@pytest.mark.parametrize('variant',VARIANTS)
def test_native_initial_pooling_real_ce_gradients_and_packet_independence(variant):
    from experiments.cvs_residual_identity.model import build as residual_build
    torch.manual_seed(17);m=build(variant);b=m.id_backbone
    torch.manual_seed(17);baseline=residual_build('residual_fusion')
    m.eval();baseline.eval();iq_check=torch.randn(2,2,256)
    with torch.no_grad():torch.testing.assert_close(m(iq_check),baseline(iq_check),atol=2e-5,rtol=2e-5)
    for name in ('t_pool','f_pool','pa_pool'):
        pool=getattr(b,name);x=torch.randn(4,pool.score.in_channels,32)
        torch.testing.assert_close(pool(x),x.mean(-1,keepdim=True))
        with torch.no_grad():pool.score.weight.normal_(0,.1)
        w=pool.weights(x);assert torch.all(w>0)
        torch.testing.assert_close(w.sum(-1),torch.ones(4,1))
        assert (w.max(-1).values/w.min(-1).values).max()<=torch.exp(torch.tensor(2.))+1e-5
        torch.testing.assert_close(pool(x),pool(x.flip(-1)),atol=1e-6,rtol=1e-5)
    optimizer=torch.optim.AdamW(m.parameters(),lr=.0002)
    m.train();iq=torch.randn(4,2,256);y=torch.arange(4)%6
    optimizer.zero_grad(set_to_none=True);F.cross_entropy(m(iq),y).backward();optimizer.step()
    for p in [b.t_proj.weight,b.f_proj.weight,b.pa_proj[0].weight,b.cls_head.gain,*b.t_pool.parameters(),*b.f_pool.parameters(),*b.pa_pool.parameters()]:
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0
    assert 164225<sum(p.numel() for p in m.parameters())<165000
    m.eval()
    with torch.no_grad():
        torch.testing.assert_close(m(iq)[:1],m(iq[:1]),rtol=2e-4,atol=2e-4)
        assert m.features(iq).shape==(4,160)
        for z in (torch.zeros(2,2,256),torch.ones(2,2,256)):assert torch.isfinite(m(z)).all()

def test_source_only_performance_selection_and_scratch_guards():
    rows=[dict(variant=v,seed=s,accuracy=.98+.001*(v==VARIANTS[1]),worst_rx=.95,parameters=164609 if v==VARIANTS[1] else 164417,macs=9715000 if v==VARIANTS[1] else 9710000,target_score=0 if v==VARIANTS[1] else 1) for v in VARIANTS for s in (2026092701,2026092702,2026092703,2026092704)]
    assert select_source_candidate(rows)['selected_variant']==VARIANTS[1]
    with pytest.raises(ValueError):select_source_candidate(rows[:-1])
    cfg=dict(method='cvs_attentive_identity',variant=VARIANTS[0],epochs=200,batch_size=128,lr=.0002,lr_min=1e-6,weight_decay=.0001,drop_last=False,augmentation=False,domain_backbone=False,extra_losses=[],selection='fixed_last_epoch',split_seed=392005)
    validate_config(cfg)
    for key,value in [('checkpoint','old.pt'),('target_inputs','target'),('augmentation',True),('domain_backbone',True)]:
        with pytest.raises(ValueError):validate_config(dict(cfg,**{key:value}))
