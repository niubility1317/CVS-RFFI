import pytest
import torch
import torch.nn.functional as F
from experiments.cvs_simplex_identity.model import build,VARIANTS
from experiments.cvs_simplex_identity.dispatch import select_source_candidate
from experiments.cvs_simplex_identity.source import validate_config
torch.set_num_threads(2)

@pytest.mark.parametrize('variant',VARIANTS)
def test_real_ce_gradients_simplex_geometry_and_packet_independence(variant):
    torch.manual_seed(19);m=build(variant);b=m.id_backbone;head=b.cls_head
    assert b.time_stability_mode=='coherence' and b.freq_stability is None
    assert b.nmfdu_gate is None and not b.mixstyle_on and not b.use_dac_path
    expected=torch.full((6,6),-.2);expected.fill_diagonal_(1.)
    m.train();x=torch.randn(4,2,256);y=torch.arange(4)%6
    opt=torch.optim.AdamW(m.parameters(),lr=.0002,weight_decay=.0001)
    for _ in range(3):
        opt.zero_grad();F.cross_entropy(m(x),y).backward()
        for p in [b.t_proj.weight,b.f_proj.weight,b.pa_proj[0].weight,head.gain,*b.time_stability.parameters()]:
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0
        if variant=='simplex_learned':
            assert head.frame.grad is not None and torch.isfinite(head.frame.grad).all() and head.frame.grad.norm()>0
            assert torch.linalg.matrix_rank(head.frame)==5
        opt.step();p=F.normalize(head.prototypes(),dim=1)
        torch.testing.assert_close(p@p.T,expected,rtol=2e-5,atol=2e-5)
    expected_count=165233 if variant=='simplex_learned' else 164433
    assert sum(p.numel() for p in m.parameters())==expected_count
    assert sum(p.numel() for p in m.parameters() if p.grad is not None)==expected_count
    m.eval()
    with torch.no_grad():
        torch.testing.assert_close(m(x)[:1],m(x[:1]),rtol=2e-4,atol=2e-4)
        for z in (torch.zeros(2,2,256),torch.ones(2,2,256)):assert torch.isfinite(m(z)).all()
        logits=m(x)
    restored=build(variant);restored.load_state_dict(m.state_dict(),strict=True);restored.eval()
    with torch.no_grad():torch.testing.assert_close(restored(x),logits,rtol=0,atol=0)

def test_matched_scratch_encoder_and_classifier_initialization():
    from experiments.cvs_coherence_identity.model import build as old
    torch.manual_seed(37);base=old('coherence_phase')
    candidates=[]
    for variant in VARIANTS:
        torch.manual_seed(37);m=build(variant);candidates.append(m)
        for key,value in base.state_dict().items():
            if key!='id_backbone.cls_head.weight':torch.testing.assert_close(m.state_dict()[key],value,rtol=0,atol=0)
    x=torch.randn(4,160)
    # Fixed prototypes are contiguous; GEMM layout causes FP32 rounding only.
    torch.testing.assert_close(candidates[0].id_backbone.cls_head.prototypes(),candidates[1].id_backbone.cls_head.prototypes(),rtol=0,atol=0)
    torch.testing.assert_close(candidates[0].id_backbone.cls_head.classify(x),candidates[1].id_backbone.cls_head.classify(x),rtol=2e-5,atol=3e-6)

def test_source_contract_and_performance_priority():
    rows=[dict(variant=v,seed=s,accuracy=.98+.001*(v==VARIANTS[0]),worst_rx=.95,parameters=165233 if v==VARIANTS[0] else 164433,macs=10000000,target_score=0 if v==VARIANTS[0] else 1) for v in VARIANTS for s in (2026092701,2026092702,2026092703,2026092704)]
    assert select_source_candidate(rows)['selected_variant']==VARIANTS[0]
    with pytest.raises(ValueError):select_source_candidate(rows[:-1])
    cfg=dict(method='cvs_simplex_identity',variant=VARIANTS[0],epochs=200,batch_size=128,lr=.0002,lr_min=1e-6,weight_decay=.0001,drop_last=False,augmentation=False,domain_backbone=False,extra_losses=[],selection='fixed_last_epoch',split_seed=392005)
    validate_config(cfg)
    for key,value in [('checkpoint','old.pt'),('target_truth','truth'),('augmentation',True),('domain_backbone',True)]:
        with pytest.raises(ValueError):validate_config(dict(cfg,**{key:value}))
