import torch
import pytest
from experiments.phase1_dura_orthogonal import source_direct_skip_probe as probe
from experiments.phase1_dura_orthogonal.direct_student_fir import DirectStudentFIR


def test_direct_path_skips_context_without_changing_student_or_logits(monkeypatch):
    class Reader:
        def features(self,x): return x.mean(-1)
        def classify_features(self,x): return torch.cat((x,x,x),dim=1)
    model=DirectStudentFIR(width=32).eval()
    x=torch.randn(2,2,256)
    calls=[]
    def context(dura,value):
        calls.append(True)
        return value*2,value*3,torch.ones(len(value))
    monkeypatch.setattr(probe,'extract_context',context)
    original=probe.run_batch(x,model,None,Reader(),True)
    fast=probe.run_batch(x,model,None,Reader(),False)
    assert len(calls)==1
    assert all(torch.equal(original[i],fast[i]) for i in (0,1,3))
    assert fast[2] is None
    with pytest.raises(ValueError,match='limited'):
        probe.run_batch(x,torch.nn.Identity(),None,Reader(),False)
