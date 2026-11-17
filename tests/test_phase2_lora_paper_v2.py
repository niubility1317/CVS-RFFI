import torch
import numpy as np
import json
import pytest

from experiments.pamf_phase2_practical import group1,group1_source,group1_target
from experiments.lora_paper_repro import phase2_source
from experiments.orthogonal_paper_repro.phase2_source import split_labeled


def test_lora_paper_profile_is_opt_in_with_author_training_budget():
    torch.set_num_threads(1)
    assert 'lora_paper_v2' not in {r['id'] for r in group1.catalog()}
    assert group1.row('lora_paper_v2')['category']=='CIL'
    model=group1_source.model_for('lora_paper_v2','cpu')
    assert model.profile=='lora_paper_v2'
    recipe=group1_source.recipe('lora_paper_v2')
    assert recipe['epochs']==1000 and recipe['source_patience']==20
    assert recipe['source_optimizer']=='RMSprop' and recipe['knn_k']==15
    assert recipe['satellite_concat'] is False and recipe['PL_start_epoch'] is None
    assert group1_target.paper_c_points({'paper_c_points':[[5,5],[10,10]]},
                                        'lora_paper_v2','MIX01',None)==[(5,5),(10,10)]


@pytest.mark.parametrize('shots',[5,10])
def test_fixed_15_neighbors_and_balanced_gallery_preserve_source(shots):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model=group1_source.model_for('lora_paper_v2','cpu').eval()
    old_y=torch.arange(6).repeat_interleave(12)
    old_f=torch.nn.functional.normalize(torch.randn(72,512),dim=1)
    x=torch.randn(5*shots,2,256)
    y=torch.arange(6,11).repeat_interleave(shots)
    before={k:v.clone() for k,v in model.state_dict().items()}
    bundle=dict(model=model,profile='lora_paper_v2',method='lora_paper_v2',
        classes=tuple(range(6)),resources={},
        native_state=dict(source_enrollment_features=old_f,source_enrollment_y=old_y))
    state=group1_target.fit_C({},bundle,x,y,tuple(range(6,11)),19,None)
    assert state['resources']['knn_k']==15
    assert state['resources']['source_samples_per_class']==shots
    assert state['resources']['actual_target_old_labels']==0
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    assert torch.bincount(state['native_state']['enrollment_y']).tolist()==[shots]*11
    with torch.no_grad():
        scores=state['score_fn'](state['model'],x[:2])
    assert scores.shape==(2,11) and torch.isfinite(scores).all()


def test_triplet_training_runs_one_epoch_with_real_short_iq_extractor():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model=group1_source.model_for('lora_paper_v2','cpu')
    labels=torch.arange(6).repeat_interleave(60)
    train,val=split_labeled(labels,17)
    spectra=torch.randn(360,1,26,6)
    before=model.network.fc.weight.detach().clone()
    history=phase2_source.train_triplets(model,spectra,labels,train,val,seed=17,max_epochs=1)
    assert len(history)==1 and np.isfinite(history[0]['training_loss'])
    assert np.isfinite(history[0]['validation_loss'])
    assert not torch.equal(before,model.network.fc.weight.detach())


def test_source_checkpoint_preserves_legal_enrollment_and_round_trips(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    class Data:
        x=torch.randn(240,2,256)
        metadata={'y':np.repeat(np.arange(6),40)}
    monkeypatch.setattr(group1_source,'source_data',lambda c,role: Data() if role=='L_s' else None)
    def fake_training(model,spectra,labels,train_idx,val_idx,**kwargs):
        assert spectra.shape==(240,1,26,6) and len(train_idx)==216 and len(val_idx)==24
        return [dict(epoch=1)]
    monkeypatch.setattr(phase2_source,'train_triplets',fake_training)
    contract=tmp_path/'contract.json'
    contract.write_text(json.dumps({'contract':'test'}),encoding='utf8')
    c=dict(run_root=str(tmp_path),source_cache='unused',source_contract=str(contract))
    out=tmp_path/'source'/'lora_paper_v2-s17'
    phase2_source.train_source(c,17,out,'cpu')
    state=group1_source.load_source(c,'lora_paper_v2',17,'cpu')
    assert state['resources']['knn_k']==15
    assert state['native_state']['source_enrollment_features'].shape==(240,512)
    assert state['provenance']['source_only'] is True


def test_legacy_source_recipe_cannot_be_relabelled_as_lora_v2(tmp_path):
    torch.set_num_threads(1)
    contract=tmp_path/'contract.json'
    contract.write_text(json.dumps({'contract':'test'}),encoding='utf8')
    folder=tmp_path/'source'/'lora_paper_v2-s17'
    folder.mkdir(parents=True)
    legacy=group1_source.model_for('lora_triplet_knn','cpu')
    torch.save(dict(schema='group1_A_v1',model=legacy.state_dict(),seed=17,
        recipe=group1_source.recipe('lora_triplet_knn'),source_contract={'contract':'test'},
        target_contact=False,ancestors=[],native_state={}),folder/'source.pt')
    c=dict(run_root=str(tmp_path),source_cache='unused',source_contract=str(contract))
    with pytest.raises(ValueError,match='Source recipe mismatch'):
        group1_source.load_source(c,'lora_paper_v2',17,'cpu')
