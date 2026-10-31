import copy
import numpy as np
import pytest
import torch
from experiments.pamf_phase3_restoration import checkpoint as ck
from experiments.pamf_phase3_restoration.train import schedule,learning_rate


def test_paired_schedule_covers_every_record_once_each_epoch_without_teacher_RNG():
    a,m=schedule(137)
    torch.manual_seed(42);torch.rand(99);np.random.seed(11);np.random.rand(99)
    b,n=schedule(137)
    assert np.array_equal(a,b) and np.array_equal(m,n)
    assert a.shape==m.shape==(30,137) and a.dtype==np.int32 and m.dtype==bool
    for row in a:assert np.array_equal(np.sort(row),np.arange(137))
    assert 0<m.sum()<m.size
    assert learning_rate(1)==.001/247 and learning_rate(247)==.001
    assert learning_rate(7410)==2e-5
    with pytest.raises(ValueError):learning_rate(7411)


@pytest.fixture
def payload():
    from experiments.pamf_phase3_restoration.model import scratch_model
    return dict(schema=ck.SCHEMA,config=ck.training_config('F0'),code_commit='a'*40,
                epoch=30,updates=7410,state_dict=scratch_model(2026092701).state_dict())


@pytest.mark.parametrize('defect',['target','OE','epoch','updates','architecture','state','dtype'])
def test_incompatible_ancestor_or_partial_state_rejected_before_deployment(monkeypatch,payload,defect):
    if defect in ('target','OE'):payload['config'][defect+'_reads']=1
    elif defect in ('epoch','updates'):payload[defect]-=1
    elif defect=='architecture':payload['config']['architecture']['name']='DURA'
    elif defect=='state':payload['state_dict'].pop('output.weight')
    else:payload['state_dict']['output.weight']=payload['state_dict']['output.weight'].half()
    monkeypatch.setattr(ck.torch,'load',lambda *a,**kw:payload)
    with pytest.raises(ValueError):ck.load_model('candidate','cpu',arm='F0',commit='a'*40)


def test_strict_final_checkpoint_roundtrip_keeps_IQ_and_cannot_be_overwritten(tmp_path):
    from experiments.pamf_phase3_restoration.model import scratch_model
    torch.set_num_threads(2);g=scratch_model(2026092701).eval()
    with torch.no_grad():g.output.weight.normal_(0,.01)
    p=tmp_path/'final.pt';ck.save_model(p,g,'IQ','a'*40,30,7410)
    loaded=ck.load_model(p,'cpu',arm='IQ',commit='a'*40)
    x=torch.randn(2,2,256)
    with torch.no_grad():torch.testing.assert_close(loaded(x),g(x),atol=0,rtol=0)
    assert all(not p.requires_grad for p in loaded.parameters())
    with pytest.raises(FileExistsError):ck.save_model(p,g,'IQ','a'*40,30,7410)
    with pytest.raises(ValueError):ck.load_model(p,'cpu',arm='F0',commit='a'*40)
