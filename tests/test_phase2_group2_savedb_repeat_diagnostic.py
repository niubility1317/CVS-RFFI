import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from experiments.pamf_phase2_practical import group2_savedb_repeat_diagnostic as diagnostic, group2_schedule


@pytest.mark.parametrize('fails', [False, True])
def test_two_repetitions_use_same_anchor_and_never_open_query(tmp_path,monkeypatch,fails):
    cap=tmp_path/'capsules/P/M/R';(cap/'splits').mkdir(parents=True)
    (cap/'manifest.json').write_text(json.dumps(dict(protocol_schema='p2_min_v1',phase2_data_status='VALIDATED_ONCE')))
    for n,split in ((0,'a'*24),(2,'b'*24)):
        (cap/'splits'/f'{split}.json').write_text(json.dumps(dict(k=1,new_count=n,split_id=split)))
    job=dict(key='p',profile='protonet_cda',seed=7,capsule='capsules/P/M/R',methods=['mopc_hr'],
             points=[[1,2]],load_B_for_mopc=str(tmp_path/'subset'))
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    monkeypatch.setattr(diagnostic,'activate',lambda rt:None)
    monkeypatch.setattr(group2_schedule,'smoke',lambda rt:dict(status='PASS',query_rows=0,truth_read=False,profiles=[1]))
    def support(_cap,split):
        size=6 if split.startswith('a') else 8
        return np.zeros((size,3),np.float32),np.arange(size),np.arange(size),{}
    monkeypatch.setattr(diagnostic,'support_only',support)
    anchor=object()
    monkeypatch.setattr(diagnostic,'load_b_state',lambda *a:dict(anchor=anchor))
    calls=[]
    def trace(prepared,x,y,seed,emit):
        assert prepared['mopc_hr'] is anchor and len(x)==len(y)==2 and y.tolist()==[6,7]
        calls.append(seed)
        if fails:
            emit(dict(event='REGISTRATION_FAILURE',completed_updates=2))
            raise FloatingPointError('nonfinite')
        model=nn.Linear(1,1)
        with torch.no_grad():model.weight.fill_(1);model.bias.fill_(0)
        return SimpleNamespace(model=model,prototypes=torch.tensor([2.]))
    monkeypatch.setattr(diagnostic,'trace_registration',trace)
    events=[];old_deterministic=torch.are_deterministic_algorithms_enabled();old_threads=torch.get_num_threads()
    old_cudnn=torch.backends.cudnn.deterministic;old_benchmark=torch.backends.cudnn.benchmark
    try:
        diagnostic.run(dict(data_root=str(tmp_path)),job,1,2,'cpu',events.append)
    finally:
        torch.use_deterministic_algorithms(old_deterministic);torch.set_num_threads(old_threads)
        torch.backends.cudnn.deterministic=old_cudnn;torch.backends.cudnn.benchmark=old_benchmark
    assert len(calls)==2 and calls[0]==calls[1]
    assert events[-1]['event']=='DIAGNOSTIC_COMPARE' and events[-1]['truth_read'] is False and events[-1]['query_rows']==0
    assert events[-1]['state_exact'] is (None if fails else True)


def test_missing_deterministic_workspace_rejected(monkeypatch):
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG',raising=False)
    with pytest.raises(ValueError,match='workspace missing'):
        diagnostic.run({}, {}, 1, 2, 'cpu', lambda event:None)
