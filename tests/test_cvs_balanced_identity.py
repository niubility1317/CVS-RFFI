import copy
import json
from pathlib import Path
import subprocess
import sys
import pytest
import torch
import torch.nn.functional as F
from experiments.cvs_balanced_identity.model import VARIANTS,build,BalancedFusion
from experiments.cvs_balanced_identity.source import validate_config
from experiments.cvs_balanced_identity.dispatch import select_source_candidate
torch.set_num_threads(2)

def test_exact_balanced_formula_and_signed_information():
    m=BalancedFusion(8,True).eval();x=torch.randn(4,17)
    torch.testing.assert_close(m(x),m.time_norm(x[:,:8])+0.5*m.frequency_norm(x[:,8:16]))
    with pytest.raises(ValueError):m(torch.randn(4,16))
    a=build('balanced_fusion');b=build('signed_balanced_fusion')
    assert isinstance(a.id_backbone.pa_proj[1],torch.nn.ReLU)
    assert isinstance(b.id_backbone.pa_proj[1],torch.nn.Identity)
    assert sum(p.numel() for p in a.parameters())<164225

@pytest.mark.parametrize('variant',VARIANTS)
def test_actual_ce_gradients_all_branches_and_frozen_inference(variant):
    m=build(variant);x=torch.randn(4,2,256);y=torch.arange(4)%6
    F.cross_entropy(m(x),y).backward()
    b=m.id_backbone
    for p in (b.t_proj.weight,b.f_proj.weight,b.pa_proj[0].weight,b.fuse.frequency_gain,b.fuse.circularity_gain,b.cls_head.gain):
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0
    assert sum(p.numel() for p in m.parameters())==sum(p.numel() for p in m.parameters() if p.grad is not None)
    m.eval();state={k:v.clone() for k,v in m.state_dict().items()}
    with torch.no_grad():
        scores=m(x);torch.testing.assert_close(scores[:1],m(x[:1]),atol=2e-4,rtol=2e-4)
        assert m.features(x).shape==(4,160)
        for z in (torch.zeros(2,2,256),torch.ones(2,2,256)):assert torch.isfinite(m(z)).all()
    for k,v in m.state_dict().items():torch.testing.assert_close(v,state[k],rtol=0,atol=0)

def test_source_guards_and_target_independent_selection():
    cfg=dict(method='cvs_balanced_identity',variant=VARIANTS[0],epochs=200,batch_size=128,lr=.0002,lr_min=1e-6,weight_decay=.0001,drop_last=False,augmentation=False,domain_backbone=False,extra_losses=[],selection='fixed_last_epoch',split_seed=392005)
    validate_config(cfg)
    for key,value in [('checkpoint','old.pt'),('target_inputs','forbidden'),('augmentation',True),('extra_losses',['x'])]:
        with pytest.raises(ValueError):validate_config(dict(cfg,**{key:value}))
    rows=[dict(variant=v,seed=s,accuracy=.98,worst_rx=.94,parameters=113665,macs=9657476,target_accuracy=1 if v==VARIANTS[1] else 0) for v in VARIANTS for s in (2026092701,2026092702,2026092703,2026092704)]
    result=select_source_candidate(rows);assert result['selected_variant']==VARIANTS[0] and result['target_score_used'] is False
    with pytest.raises(ValueError):select_source_candidate(rows[:-1])
    subprocess.run([sys.executable,'-m','experiments.cvs_balanced_identity.source','--help'],cwd=Path(__file__).resolve().parents[1],capture_output=True,check=True)
