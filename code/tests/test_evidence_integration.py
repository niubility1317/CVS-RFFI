import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from post_stage_common import build_baseline_model
from cvsrffi.evidence_head import EvidenceConfig
from cvsrffi.evidence_head_training import initialize_from_source,supervised_evidence_loss,mechanism_manifest
from cvsrffi.evidence_observation import normalize_observed


def model_args(variant=None,layout="joint"):
    cfg={} if variant is None else dict(variant=variant,layout=layout,covariance_rank=0 if variant=="H1" else 2)
    return SimpleNamespace(num_classes=3,num_domains=2,model_size="M",dataset="wisig",input_len=256,
            sample_rate_hz=25e6,model_variant="lite_d",branch_ablation="no_dac",domain_branch_ablation="no_stats",
            evidence_config="" if variant is None else json.dumps(cfg))


def source_batch():
    gen=torch.Generator().manual_seed(139)
    x=torch.randn(9,2,256,generator=gen)
    x[::3]=torch.cumsum(x[::3],-1)/5
    x[1::3,:,::4]*=2
    return x,torch.arange(9)%3


@pytest.mark.parametrize("variant",["H1","H2","H3","H4","H5"])
def test_real_core90_routes_backprop_reload(variant,tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(3)
    args=model_args(variant)
    model=build_baseline_model(args,torch.device("cpu"))
    x,y=source_batch()
    init=initialize_from_source(model,[(x,y)])
    assert init["quality_calibrated"] and init["calibration_pairs"]==9
    model.eval()
    # Fast inference and auxiliary forward must use exactly the same new head.
    with torch.no_grad():
        out=model(x,y_tx=None,return_aux=True)
        fast=model(x,return_aux=False)
        torch.testing.assert_close(fast,out["tx_logits"])
        torch.testing.assert_close(fast,torch.cat([model(row[None],return_aux=False) for row in x]),rtol=2e-4,atol=2e-4)
    model.train()
    out=model(x,y_tx=y,return_aux=True)
    extra={"physical_sample_id":[f"source-{i}" for i in range(9)]}
    loss,stats=supervised_evidence_loss(model,out,y,extra,len(y))
    loss=loss+F.cross_entropy(out["tx_logits"],y)
    before={n:p.detach().clone() for n,p in model.evidence_head.named_parameters() if p.requires_grad and p.numel()}
    opt=torch.optim.AdamW(model.parameters(),lr=.002)
    loss.backward()
    assert torch.isfinite(loss)
    for n,p in model.evidence_head.named_parameters():
        if p.grad is not None: assert torch.isfinite(p.grad).all(),n
    assert model.evidence_head.response.mean.grad.abs().sum()>0
    assert model.evidence_head.raw_diagonal.grad.abs().sum()>0
    if variant!="H1": assert model.evidence_head.factor.grad.abs().sum()>0
    if variant in {"H3","H4","H5"}:
        assert model.evidence_head.response.class_slopes.grad.abs().sum()>0
        assert stats["evidence/state_covariance_energy"]>0
    if variant in {"H4","H5"}: assert stats["evidence/support_queries"]>0
    if variant=="H5": assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.evidence_head.pair.parameters())
    opt.step()
    assert any(not torch.equal(p,before[n]) for n,p in model.evidence_head.named_parameters() if n in before)
    model.eval()
    with torch.no_grad(): final=model(x,return_aux=False)
    path=tmp_path/"model.pt"
    torch.save({"model":model.state_dict(),"baseline_args":vars(args)},path)
    loaded=torch.load(path,weights_only=True)
    restored=build_baseline_model(SimpleNamespace(**loaded["baseline_args"]),torch.device("cpu"))
    restored.load_state_dict(loaded["model"],strict=True); restored.eval()
    with torch.no_grad(): torch.testing.assert_close(final,restored(x,return_aux=False))
    # Pseudolabel path may change representation but not response/variance/expert.
    model.train(); model.zero_grad(set_to_none=True)
    u=model(x,return_aux=True)
    F.cross_entropy(u["tx_logits"],y).backward()
    assert all(p.grad is None or not p.grad.any() for p in model.evidence_head.parameters())


def test_h0_blocks_mask_and_strict_config():
    torch.set_num_threads(2)
    model=build_baseline_model(model_args(),torch.device("cpu"));model.eval()
    assert not hasattr(model,"evidence_head")
    x,y=source_batch()
    with torch.no_grad(): torch.testing.assert_close(model(x),model(x,return_aux=True)["tx_logits"])
    block=build_baseline_model(model_args("H2","blocks"),torch.device("cpu"))
    initialize_from_source(block,[(x,y)])
    block.eval()
    with torch.no_grad(): assert block(x).shape==(9,3)
    assert block.evidence_head.feature_dim==480
    z=torch.tensor([[float("nan"),3.,4.,2.]])
    mask=torch.tensor([[False,True,True,True]])
    got=normalize_observed(z,mask,(2,2))
    z[0,0]=1e20
    torch.testing.assert_close(got,normalize_observed(z,mask,(2,2)))
    with pytest.raises(TypeError): EvidenceConfig.parse({"varaint":"H3"})
    with pytest.raises(ValueError): EvidenceConfig(variant="H1",covariance_rank=2)
    fresh=build_baseline_model(model_args("H2"),torch.device("cpu"))
    with pytest.raises(RuntimeError): fresh(x)
    with pytest.raises(ValueError): initialize_from_source(fresh,[(x,y)],role="V")
