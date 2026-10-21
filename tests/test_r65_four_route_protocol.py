import copy
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from experiments.pamf_r65_four_route import design as d
from experiments.pamf_r65_four_route import runtime,evaluation,training


@pytest.mark.parametrize('field,value',[
    ('training_TX_labels_read',1),('training_F_access',True),('F_weights_read',1),
    ('F_outputs_read',1),('target_training',True),('V_training',True),
    ('parent_row','prune_after5'),('data_contract','different_split'),
    ('updates',4939),('epoch',21),('arm','unknown')])
def test_checkpoint_rejects_invalid_source_contract(field,value):
    payload=runtime.metadata('stage2',20,4940,{},'a'*40)
    payload[field]=value
    with pytest.raises(ValueError):runtime.validate_payload(payload)


def test_unknown_route_rejected_before_io(tmp_path):
    with pytest.raises(ValueError):runtime.initialize('unknown',tmp_path,'cpu')
    with pytest.raises(ValueError):runtime.validate_parent(tmp_path)


def test_incomplete_training_rejected_before_target_io(monkeypatch,tmp_path):
    calls=[]
    def read(path):
        calls.append(path)
        return dict(status='SOURCE_TRAINED',arm='stage2',epochs=19,updates=4693)
    monkeypatch.setattr(evaluation,'read_json',read)
    args=SimpleNamespace(output=tmp_path/d.RUN/'stage2',kind=d.KIND)
    with pytest.raises(ValueError):evaluation.reference(args,'target')
    assert calls==[args.output/'training_complete.json']


@pytest.mark.parametrize('arm',d.ARMS)
def test_alignment_and_teacher_clean_stop_gradients(arm):
    teacher=[torch.full((2,2,256),float(i+1),requires_grad=True) for i in range(3)]
    student=[teacher[i].detach().clone().requires_grad_(True) for i in d.STAGE_TARGETS[arm]]
    clean=torch.ones((2,2,256),requires_grad=True)
    loss,terms=training.objective(student,teacher,clean,1,arm)
    assert torch.equal(terms['teacher_stage_nmse'],torch.zeros(2))
    loss.mean().backward()
    assert all(t.grad is None for t in teacher)
    assert clean.grad is None
    assert student[-1].grad is not None


@pytest.mark.parametrize('arm',d.ARMS)
def test_portable_zero_update_checkpoint_preserves_aliases_and_outputs(tmp_path,arm):
    torch.set_num_threads(2);torch.manual_seed(42)
    model=runtime.build(arm).eval();x=torch.randn(1,2,256)*.1
    path=tmp_path/'smoke.pt'
    runtime.save_checkpoint(path,model,0,0,'a'*40)
    fresh,payload=runtime.load_checkpoint(path,'cpu')
    with torch.no_grad():
        assert torch.equal(model(x),fresh(x))
    assert payload['updates']==0
    runtime.validate_aliases(fresh,payload['model'])


def test_four_routes_and_frozen_budget_are_not_score_selected():
    value=d.configuration()
    assert tuple(value['arms'])==d.ARMS and len(d.ARMS)==4
    assert value['updates']==4940 and value['epochs']==20
    assert value['max_own_GPU_workers']==2 and not value['target_feedback']
    assert d.STAGE_TARGETS['stage2']==(1,2)
    assert all(d.GPUS[a] in (0,1) for a in d.ARMS)
