import numpy as np
import pytest
import torch
from experiments.phase1_dura_orthogonal import signal_mix_gate_ground as ground
from experiments.phase1_dura_orthogonal.signal_mix_gate_train import select_oof_ids


def test_source_fit_excludes_direct_parent_pilot_and_protected_days():
    rx=np.array([1,3,4,4,6,8,1,1]);day=np.array([1,1,3,2,3,1,1,1]);roles=np.array(['L_s','U_s','L_s','U_s','U_s','L_s','U_s','L_s'])
    assert select_oof_ids(rx,day,roles,np.array([0,1]),np.array([6]),expected_size=3).tolist()==[3,5,7]
    roles[5]='V'
    with pytest.raises(ValueError):select_oof_ids(rx,day,roles,np.array([0,1]),np.array([6]),expected_size=3)


def test_direct_parent_and_dura_are_mixed_before_reader(monkeypatch):
    x=torch.zeros(2,2,256);anchor=torch.ones_like(x);direction=torch.zeros_like(x);variance=torch.ones(2)
    monkeypatch.setattr(ground,'extract_context',lambda dura,value:(anchor,direction,variance))
    parent=lambda received,a,d,v:torch.full_like(received,3.)
    gate=lambda received,a,d,p,v:a+.25*(p-a)
    class Reader:
        def features(self,value):return value.mean((1,2))
        def classify_features(self,value):return torch.stack((value,2-value,value-1,value-2,value-3,value-4),1)
    out=ground.decisions(x,parent,None,Reader(),gate=gate,expected_mode='fixed',inference_mode='fixed')
    assert [v.tolist() for v in out]==[[1,1],[0,0],[0,0]]
    with pytest.raises(ValueError):ground.decisions(x,parent,None,Reader(),gate=gate,expected_mode='fixed',inference_mode='other')
