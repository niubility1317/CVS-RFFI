import json
import numpy as np
import pytest
import torch
from torch import nn

from experiments.pamf_phase2_practical import group2_savedb_diagnostic as diagnostic
from experiments.pamf_phase2_practical import group2_schedule
from experiments.pamf_phase2_practical.group2_cil_bridge import AdaptedClassifier, make_anchor
from experiments.pamf_phase2_practical.group2_cross_fit import register_c
from experiments.pamf_phase2_practical.group2_dadda_mopc_diagnostic import trace_registration
from experiments.pamf_phase2_practical.vendor_native.comparison_suite import registration


def test_forward_hooks_preserve_training_and_rng():
    previous_threads=torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(21)
            model=AdaptedClassifier(nn.Linear(3,4),nn.Linear(4,6),None)
            prepared={'mopc_hr':make_anchor(model,torch.randn(6,3),torch.arange(6),method='mopc_hr')}
            x,y=torch.randn(10,3),torch.arange(6,16)
            baseline=register_c(prepared,x,y,seed=42,methods=('mopc_hr',))['mopc_hr']
            rng=torch.get_rng_state().clone()
            events=[]
            with diagnostic.observe_support_forward(events.append):
                observed=trace_registration(prepared,x,y,42,events.append)
            assert torch.equal(torch.get_rng_state(),rng)
            assert all(torch.equal(value,observed.model.state_dict()[key])
                       for key,value in baseline.model.state_dict().items())
            assert torch.equal(baseline.prototypes,observed.prototypes)
            assert not any(e['event']=='FIRST_FORWARD_NONFINITE' for e in events)
    finally:
        torch.set_num_threads(previous_threads)


def test_first_nonfinite_leaf_is_recorded_and_hooks_removed(monkeypatch):
    model=nn.Sequential(nn.Linear(2,2))
    with torch.no_grad(): model[0].weight.fill_(float('inf'))
    monkeypatch.setattr(registration,'_logits',lambda m,_method,x:m(x))
    events=[]
    with diagnostic.observe_support_forward(events.append):
        result=registration._logits(model,'mopc_hr',torch.ones(1,2))
    assert not torch.isfinite(result).all()
    assert len(events)==1 and events[0]['layer']=='0' and events[0]['boundary']=='output'
    assert not model[0]._forward_hooks and not model[0]._forward_pre_hooks


def test_diagnostic_reads_only_support_and_rejects_mismatched_old_positions(tmp_path,monkeypatch):
    cap=tmp_path/'capsules/P/M/R';(cap/'splits').mkdir(parents=True)
    (cap/'manifest.json').write_text(json.dumps(dict(protocol_schema='p2_min_v1',phase2_data_status='VALIDATED_ONCE')))
    for n,split in ((0,'a'*24),(2,'b'*24)):
        (cap/'splits'/f'{split}.json').write_text(json.dumps(dict(k=1,new_count=n,split_id=split)))
    job=dict(key='p',profile='protonet_cda',seed=7,capsule='capsules/P/M/R',methods=['mopc_hr'],
             points=[[1,2]],load_B_for_mopc=str(tmp_path/'subset'))
    monkeypatch.setattr(diagnostic,'activate',lambda rt:None)
    monkeypatch.setattr(group2_schedule,'smoke',lambda rt:dict(status='PASS',query_rows=0,truth_read=False,profiles=[1]))
    def support(_cap,split):
        size=6 if split.startswith('a') else 8
        return np.zeros((size,3),np.float32),np.arange(size),np.arange(size),{}
    monkeypatch.setattr(diagnostic,'support_only',support)
    monkeypatch.setattr(diagnostic,'load_b_state',lambda *a:dict(anchor='saved'))
    seen=[]
    def trace(prepared,x,y,seed,emit):
        assert prepared['mopc_hr']=='saved' and len(x)==len(y)==2 and y.tolist()==[6,7]
        seen.append(True)
    monkeypatch.setattr(diagnostic,'trace_registration',trace)
    events=[]
    diagnostic.run(dict(data_root=str(tmp_path)),job,1,2,'cpu',events.append)
    assert seen==[True] and all(event['query_rows']==0 and event['truth_read'] is False for event in events)
    def mismatch(cap,split):
        x,y,ids,meta=support(cap,split)
        if split.startswith('b'):ids[0]=99
        return x,y,ids,meta
    monkeypatch.setattr(diagnostic,'support_only',mismatch)
    with pytest.raises(ValueError,match='old support positions differ'):
        diagnostic.run(dict(data_root=str(tmp_path)),job,1,2,'cpu',events.append)
