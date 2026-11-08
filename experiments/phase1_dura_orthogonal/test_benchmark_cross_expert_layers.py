import torch
from experiments.phase1_dura_orthogonal import benchmark_cross_expert_layers as probe
from experiments.phase1_dura_orthogonal.cross_expert_refiner import CrossExpertRefiner


def test_hooks_preserve_nonzero_model_output_and_are_removed(monkeypatch):
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    torch.manual_seed(42);model=CrossExpertRefiner(width=8).eval()
    with torch.no_grad():
        for head in (model.time_out,model.freq_out):
            torch.nn.init.normal_(head.weight,std=.001);torch.nn.init.normal_(head.bias,std=.001)
    x=torch.randn(3,2,256);a=torch.randn_like(x);d=torch.randn_like(x);p=torch.randn_like(x);v=torch.rand(3)
    args=(x,a,d,p,v)
    with torch.no_grad():
        reference=model(*args);actual,rows,elapsed=probe.profiled_forward(model,args);after=model(*args)
    assert probe.same_bits(reference,actual) and probe.same_bits(reference,after)
    assert not probe.same_bits(reference,a+.5*(p-a))
    assert {r['kind'] for r in rows}=={'Conv1d','GroupNorm','GELU'}
    assert all(r['seconds']>=0 for r in rows) and elapsed>=0
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
