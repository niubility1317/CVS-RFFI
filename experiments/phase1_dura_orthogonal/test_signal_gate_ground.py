import pytest
import torch
from experiments.phase1_dura_orthogonal import signal_gate_ground as module


def test_decisions_use_matching_gate_and_reader(monkeypatch):
    x=torch.zeros(2,2,256)
    anchor=torch.ones_like(x)
    direction=torch.zeros_like(x)
    variance=torch.ones(2)
    monkeypatch.setattr(module,'extract_context',lambda dura,value:(anchor,direction,variance))
    parent=lambda received,a,d,v:a+2
    gate=lambda received,a,d,p,v:a+.25*(p-a)
    class Reader:
        def features(self,value):return value.mean((1,2))
        def classify_features(self,values):
            return torch.stack((values,2-values,values-1,values-2,values-3,values-4),1)
    reader=Reader()
    out=module.decisions(x,parent,None,reader,gate=gate,expected_mode='fixed',inference_mode='fixed')
    assert [v.tolist() for v in out]==[[1,1],[0,0],[0,0]]
    with pytest.raises(ValueError):
        module.decisions(x,parent,None,reader,gate=gate,expected_mode='fixed',inference_mode='wrong')
