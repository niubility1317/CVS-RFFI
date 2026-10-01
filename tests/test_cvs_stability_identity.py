import pytest
import torch
import torch.nn.functional as F
from experiments.cvs_stability_identity.model import build,VARIANTS
from experiments.cvs_stability_identity.dispatch import select_source_candidate
from experiments.cvs_stability_identity.source import validate_config
torch.set_num_threads(2)

@pytest.mark.parametrize('variant',VARIANTS)
def test_physical_cues_active_real_ce_gradients_and_packet_independence(variant):
    m=build(variant);b=m.id_backbone
    assert b.time_stability_mode=='phase_delta' and (b.freq_stability is not None)==(variant=='phase_dsq')
    assert b.emb_dim==160 and b.time_stability_channels==8 and b.freq_stability_channels==4
    assert b.nmfdu_gate is None and b.crra_time is None and not b.mixstyle_on and not b.use_dac_path
    m.train();x=torch.randn(4,2,256);y=torch.arange(4)%6
    F.cross_entropy(m(x),y).backward()
    extra=list(b.time_stability.parameters())+([] if b.freq_stability is None else list(b.freq_stability.parameters()))
    for p in [*extra,b.t_proj.weight,b.f_proj.weight,b.pa_proj[0].weight,b.cls_head.gain]:
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0
    assert 164225<sum(p.numel() for p in m.parameters())<170000
    m.eval()
    with torch.no_grad():
        torch.testing.assert_close(m(x)[:1],m(x[:1]),rtol=2e-4,atol=2e-4)
        assert m.features(x).shape==(4,160)
        for z in (torch.zeros(2,2,256),torch.ones(2,2,256)):assert torch.isfinite(m(z)).all()

def test_stem_phase_rotation_invariance_and_frequency_constant_residual():
    m=build('phase_dsq');phase=m.id_backbone.time_stability.eval();dsq=m.id_backbone.freq_stability.eval()
    x=torch.randn(2,48,64);r,i=x.chunk(2,1);a=torch.tensor(.71)
    rotated=torch.cat((a.cos()*r-a.sin()*i,a.sin()*r+a.cos()*i),1)
    torch.testing.assert_close(phase(x),phase(rotated),rtol=5e-5,atol=5e-5)
    f=torch.randn(2,4,32);shift=torch.tensor([1.,-.5,.2,2.]).view(1,4,1)
    torch.testing.assert_close(dsq(f),dsq(f+shift),rtol=5e-5,atol=5e-5)

def test_source_performance_priority_and_scratch_guards():
    rows=[dict(variant=v,seed=s,accuracy=.98+.001*(v==VARIANTS[1]),worst_rx=.95,parameters=166000 if v==VARIANTS[1] else 165000,macs=10020000 if v==VARIANTS[1] else 10000000,target_score=0 if v==VARIANTS[1] else 1) for v in VARIANTS for s in (2026092701,2026092702,2026092703,2026092704)]
    assert select_source_candidate(rows)['selected_variant']==VARIANTS[1]
    with pytest.raises(ValueError):select_source_candidate(rows[:-1])
    cfg=dict(method='cvs_stability_identity',variant=VARIANTS[0],epochs=200,batch_size=128,lr=.0002,lr_min=1e-6,weight_decay=.0001,drop_last=False,augmentation=False,domain_backbone=False,extra_losses=[],selection='fixed_last_epoch',split_seed=392005)
    validate_config(cfg)
    for key,value in [('checkpoint','old.pt'),('target_inputs','target'),('augmentation',True),('domain_backbone',True)]:
        with pytest.raises(ValueError):validate_config(dict(cfg,**{key:value}))
