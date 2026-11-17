import torch
import numpy as np
import pytest

from experiments.pamf_phase2_practical import group1, group1_source, group1_target
from experiments.orthogonal_paper_repro import phase2_source
from experiments.orthogonal_paper_repro.increment import register_session


def test_paper_profile_is_explicit_and_source_architecture_matches_reproduction():
    torch.set_num_threads(1)
    assert 'orthogonal_paper_v2' not in {r['id'] for r in group1.catalog()}
    assert group1.row('orthogonal_paper_v2')['category'] == 'CIL'
    model = group1_source.model_for('orthogonal_paper_v2', 'cpu')
    assert tuple(model.pseudo_targets.shape) == (200, 256)
    assert tuple(model.classifier_weight.shape) == (6, 256)
    recipe = group1_source.recipe('orthogonal_paper_v2')
    assert recipe['epochs'] == 100 and recipe['satellite_concat'] is False
    assert recipe['early_stopping'] is True
    assert recipe['PL_start_epoch'] is None and recipe['source_validation_fraction'] == 0.1


def test_paper_registration_accepts_k10_and_preserves_old_model():
    torch.set_num_threads(1)
    torch.manual_seed(11)
    model = group1_source.model_for('orthogonal_paper_v2', 'cpu').eval()
    before = {k:v.clone() for k,v in model.state_dict().items()}
    x = torch.randn(50, 2, 256)
    y = torch.arange(6,11).repeat_interleave(10)
    logs=[]
    bundle=dict(model=model,profile='orthogonal_paper_v2',method='orthogonal_paper_v2',
                classes=tuple(range(6)),resources={})
    state=group1_target.fit_C({},bundle,x,y,tuple(range(6,11)),19,logs.append)
    assert state['classes'] == tuple(range(11))
    assert 1 <= len(logs) <= 50
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    assert torch.equal(state['model'].classifier_weight[:6],before['classifier_weight'])
    with torch.no_grad():
        scores=state['score_fn'](state['model'],x[:2])
    assert scores.shape == (2,11) and torch.isfinite(scores).all()


def test_k5_registration_matches_fixed_reproduction():
    torch.set_num_threads(1)
    torch.manual_seed(41)
    model=group1_source.model_for('orthogonal_paper_v2','cpu').eval()
    x=torch.randn(25,2,256)
    y=torch.arange(6,11).repeat_interleave(5)
    expected,history=register_session(model.encoder,model.classifier_weight,x,y)
    bundle=dict(model=model,profile='orthogonal_paper_v2',method='orthogonal_paper_v2',
                classes=tuple(range(6)),resources={})
    state=group1_target.fit_C({},bundle,x,y,tuple(range(6,11)),17,None)
    assert len(history)==state['resources']['C_epochs']
    assert torch.equal(expected,state['model'].classifier_weight)


def test_source_split_is_stratified_deterministic_and_disjoint():
    labels=torch.arange(6).repeat_interleave(20)
    train,validation=phase2_source.split_labeled(labels,19)
    assert len(train)==108 and len(validation)==12
    assert not set(train.tolist()).intersection(validation.tolist())
    assert all(int((labels[validation]==i).sum())==2 for i in range(6))
    assert torch.equal(train,phase2_source.split_labeled(labels,19)[0])


def test_paper_points_do_not_silently_expand_or_accept_other_inputs():
    job={'paper_c_points':[[5,5],[10,10]]}
    assert group1_target.paper_c_points(job,'orthogonal_paper_v2','MIX01',None)==[(5,5),(10,10)]
    for bad in ({'paper_c_points':[[5,5],[5,5]]},
                {'paper_c_points':[[5,20]]},
                {'paper_c_points':[[5,5]],'scope':'pilot_main'},
                {'paper_c_points':[[5,5]],'satellite_c_points':[[5,5]]}):
        with pytest.raises(ValueError):
            group1_target.paper_c_points(bad,'orthogonal_paper_v2','MIX01',None)
    with pytest.raises(ValueError):
        group1_target.paper_c_points(job,'orthogonal','MIX01',None)
    with pytest.raises(ValueError):
        group1_target.paper_c_points(job,'orthogonal_paper_v2','MIX01','clean-root')


def test_source_training_checkpoint_round_trip(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    import json
    class Data:
        x=torch.randn(120,2,256)
        metadata={'y':np.repeat(np.arange(6),20)}
    monkeypatch.setattr(group1_source,'source_data',lambda c,role: Data() if role=='L_s' else None)
    def fake_train(encoder,train_x,train_y,val_x,val_y,**kwargs):
        assert len(train_y)==108 and len(val_y)==12 and kwargs['num_classes']==6
        targets=torch.randn(200,256)
        return dict(targets=targets,base_weights=targets[:6].clone(),best_epoch=1),[dict(epoch=1)]
    monkeypatch.setattr(phase2_source,'train_base',fake_train)
    contract=tmp_path/'contract.json'
    contract.write_text(json.dumps({'contract':'test'}),encoding='utf8')
    c=dict(run_root=str(tmp_path),source_cache='unused',source_contract=str(contract))
    out=tmp_path/'source'/'orthogonal_paper_v2-s17'
    phase2_source.train_source(c,17,out,'cpu')
    state=group1_source.load_source(c,'orthogonal_paper_v2',17,'cpu')
    assert state['model'].classifier_weight.shape==(6,256)
    assert state['provenance']['source_only'] is True
