import pytest
import torch
import torch.nn.functional as F
from experiments.cvs_interaction_identity.model import build,VARIANTS
from experiments.cvs_interaction_identity.dispatch import select_source_candidate
from experiments.cvs_interaction_identity.source import validate_config
torch.set_num_threads(2)

@pytest.mark.parametrize('variant',VARIANTS)
def test_direct_path_identity_at_initialization_and_ce_interaction_updates(variant):
    m=build(variant);b=m.id_backbone;fusion=b.fuse.eval();x=torch.randn(4,321)
    t=fusion.time_norm(x[:,:160]);f=fusion.frequency_norm(x[:,160:320])
    torch.testing.assert_close(fusion(x),t+0.5*f,rtol=1e-6,atol=1e-6)
    optimizer=torch.optim.AdamW(m.parameters(),lr=.0002)
    m.train();iq=torch.randn(4,2,256);y=torch.arange(4)%6
    for step in range(2):
        optimizer.zero_grad(set_to_none=True);F.cross_entropy(m(iq),y).backward();optimizer.step()
    for p in [b.t_proj.weight,b.f_proj.weight,b.pa_proj[0].weight,*b.fuse.parameters(),b.cls_head.gain]:
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0
    assert sum(p.numel() for p in m.parameters())<164225
    m.eval()
    with torch.no_grad():
        torch.testing.assert_close(m(iq)[:1],m(iq[:1]),rtol=2e-4,atol=2e-4)
        assert m.features(iq).shape==(4,160)
        for z in (torch.zeros(2,2,256),torch.ones(2,2,256)):assert torch.isfinite(m(z)).all()

def test_performance_priority_and_scratch_source_contract():
    rows=[dict(variant=v,seed=s,accuracy=.98+.001*(v==VARIANTS[0]),worst_rx=.95,parameters=130000 if v==VARIANTS[0] else 114000,macs=9700000 if v==VARIANTS[0] else 9660000,target_score=0 if v==VARIANTS[0] else 1) for v in VARIANTS for s in (2026092701,2026092702,2026092703,2026092704)]
    assert select_source_candidate(rows)['selected_variant']==VARIANTS[0]
    with pytest.raises(ValueError):select_source_candidate(rows[:-1])
    cfg=dict(method='cvs_interaction_identity',variant=VARIANTS[0],epochs=200,batch_size=128,lr=.0002,lr_min=1e-6,weight_decay=.0001,drop_last=False,augmentation=False,domain_backbone=False,extra_losses=[],selection='fixed_last_epoch',split_seed=392005)
    validate_config(cfg)
    for key,value in [('checkpoint','old.pt'),('target_inputs','target'),('augmentation',True),('domain_backbone',True)]:
        with pytest.raises(ValueError):validate_config(dict(cfg,**{key:value}))
